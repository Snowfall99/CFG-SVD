import path2data_w
from typing import Union, Tuple
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, TopKPooling, GraphConv
import torch_geometric.nn
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
import torch.nn.functional as F
import random
import numpy as np
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import time
# 神经网络模型加载
# GCN Layer
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        # 聚集方案：add
        super(GCNConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # x, edge_index = data.x, data.edge_index

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(128, 128)
        #self.conv1 = GraphConv(256, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GCNConv(128, 128)
        #self.conv2 = GraphConv(256, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GCNConv(128, 128)
        #self.conv3 = GraphConv(256, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.conv4 = GCNConv(128, 128)
        self.pool4 = TopKPooling(128, ratio=0.8)
        self.conv5 = GCNConv(128, 128)
        self.pool5 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 2)

        self.readout = Seq(Linear(128, 64),
                           ReLU(),
                           Linear(64, 2))

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv5(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)
        x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4 + x5

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.softmax(self.lin3(x), dim=-1)

        return x


# 加载准备好的数据集
datapath = r"F:\testcases.tar\test"
dataset = path2data_w.loadPath2DataSet(datapath)
trainloader = DataLoader(dataset=dataset,batch_size=1,shuffle=True)
# 加载模型
model_list =  []
model_400  = torch.load(r'D:\torch-geometric\test\CEW400.pt')
model_list.append(model_400)
model_401  = torch.load(r'D:\torch-geometric\test\CWE401.pt')
model_list.append(model_400)
model_590  = torch.load(r'D:\torch-geometric\test\CWE590.pt')
model_list.append(model_590)
model_762  = torch.load(r'D:\torch-geometric\test\CWE762.pt')
model_list.append(model_762)
model_789  = torch.load(r'D:\torch-geometric\test\CWE789.pt')
model_list.append(model_789)

# 设置阈值
thre = 0.6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 记录漏洞信息————即模型预测值
All_Bugs = []   # 实际的预测值
for data in trainloader:
    Bugs = {"CWE400":0,"CWE401":1,"CWE590":2,"CWE762":3,"CWE789":4}
    for key in Bugs.keys():
        model = Net().to(device)
        model.load_state_dict(model_list[Bugs[key]])
        model.eval()
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        pred = model(x, edge_index, batch)
        Bugs[key] = pred
    All_Bugs.append(Bugs)

# 通过阈值判断漏洞类型
All_Pre_Bugs = [] # 通过阈值比较过的的结果
for Bugs in All_Bugs:
    Pre_Bugs = {"CWE400": False, "CWE401": False, "CWE590": False, "CWE762": False, "CWE789": False}
    for key in Bugs.keys():
        value = Bugs[key].detach().numpy().tolist()
        if value[0][0] >= thre:
            Pre_Bugs[key] = True









