import path2data1
from typing import Union, Tuple
import torch
import torch.nn as nn
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
import datetime
import matplotlib.pyplot as plt

# generate vulner models
def genModel(number, set):
    modelSet = []
    num = 0
    for i in range(len(set)):
        if set[i].y[0].item() == 1:
            modelSet.append(set[i])
            num += 1
            if num == number:
                return modelSet
    return modelSet

#change trainsets
def transformSet(set):
    res = []
    for i in range(len(set)//2-1):
        twoData = []
        twoData.append(set[2*i])
        twoData.append(set[2*i+1])
        res.append(twoData)
    return res

good_label = 0
bad_label = 0

datapath = r"/home/chenzx/testcases/CWE590_Free_Memory_Not_on_Heap"

dataset = path2data1.loadPath2DataSet(datapath)
random.shuffle(dataset)

modelSet = genModel(100, dataset)

lenDataset = len(dataset)
lenTrainset = int(0.7*lenDataset)
lenValidset = int(0.2*lenDataset)
lenTestset = lenDataset - lenTrainset - lenValidset
print("数据集总量：%d 训练集：%d 验证集：%d 测试集：%d" % (lenDataset, lenTrainset, lenValidset, lenTestset))
trainSet = dataset[:lenTrainset]
trainSet = transformSet(trainSet)
# 验证集
validateSet = dataset[lenTrainset:lenTrainset+lenValidset]
# 测试集
testSet = dataset[lenTrainset+lenValidset:]

sadataset = []
for i in range(len(testSet)):
    for j in range(len(modelSet)):
        twoData = []
        twoData.append(dataset[i])
        twoData.append(modelSet[j])
        sadataset.append(twoData)

# 加载训练集
trainloader = DataLoader(dataset=trainSet,batch_size=1, shuffle=True)
testloader = DataLoader(dataset=testSet, batch_size=1, shuffle=True)
modelloader = DataLoader(dataset=modelSet, batch_size=1, shuffle=True)
sadataloader = DataLoader(dataset=sadataset, batch_size=1, shuffle=False)

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

        self.conv1 = GCNConv(256, 256)
        # self.conv1 = GraphConv(128, 128)
        self.pool1 = TopKPooling(256, ratio=0.8)
        self.conv2 = GCNConv(256, 256)
        # self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(256, ratio=0.8)
        self.conv3 = GCNConv(256, 256)
        # self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(256, ratio=0.8)
        self.conv4 = GCNConv(256, 256)
        self.pool4 = TopKPooling(256, ratio=0.8)
        self.conv5 = GCNConv(256, 256)
        self.pool5 = TopKPooling(256, ratio=0.8)

        self.convAtt1 = torch.nn.Conv1d(in_channels=512, out_channels=64, kernel_size=1, stride=2)
        self.poolAtt1 = torch.nn.MaxPool1d(kernel_size=1, stride=2)
        self.convAtt2 = torch.nn.Conv1d(64, 16, kernel_size=1, stride=2)
        self.poolAtt2 = torch.nn.MaxPool1d(kernel_size=1, stride=2)
        self.convAtt3 = torch.nn.Conv1d(16, 2, kernel_size=1, stride=2)
        self.poolAtt3 = torch.nn.MaxPool1d(kernel_size=1, stride=2)

        self.convAtt4 = torch.nn.Conv1d(2, 16, kernel_size=1, stride=2)
        self.poolAtt4 = torch.nn.MaxPool1d(kernel_size=1, stride=2)
        self.convAtt5 = torch.nn.Conv1d(16, 64, kernel_size=1, stride=2)
        self.poolAtt5 = torch.nn.MaxPool1d(kernel_size=1, stride=2)
        self.convAtt6 = torch.nn.Conv1d(64, 512, kernel_size=1, stride=2)
        self.poolAtt6 = torch.nn.MaxPool1d(kernel_size=1, stride=2)

        self.lin1 = torch.nn.Linear(512, 128)
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

        x = x1 + x2 + x3

        x = 1/3 * x 
        
        sx = x

        x = x.unsqueeze(dim=2)
        # attention层
        x = F.relu(self.convAtt1(x))
        x = self.poolAtt1(x)
        x = F.relu(self.convAtt2(x))
        x = self.poolAtt2(x)
        x = F.relu(self.convAtt3(x))
        x = self.poolAtt3(x)

        x = F.relu(self.convAtt4(x))
        x = self.poolAtt4(x)
        x = F.relu(self.convAtt5(x))
        x = self.poolAtt5(x)
        x = F.relu(self.convAtt6(x))
        x = self.poolAtt6(x)

        x = x.squeeze()

        x = (x + 1) * sx

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)    
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive, euclidean_distance





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = ContrastiveLoss()


# generate model vector
def genVec(set):
    vecSet = []
    for data in modelloader:
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        output = model(x, edge_index, batch)
        vecSet.append(output)
    return vecSet

modelVec = genVec(modelloader)

def train(epoch):
    model.train()
    loss_all = 0

    for data in trainloader:
        data1 = data[0].to(device)
        optimizer.zero_grad()
        x, edge_index, batch = data1.x, data1.edge_index, data1.batch
        output1 = model(x, edge_index, batch)

        data2 = data[1].to(device)
        optimizer.zero_grad()
        x, edge_index, batch = data2.x, data2.edge_index, data2.batch
        output2 = model(x, edge_index, batch)

        loss, distance = criterion(output1,output2,int(data1.y[0]!=data2.y[0]))
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

    return loss_all / len(trainSet)

def myTest(loader, level):
    model.eval()
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    num = 0
    total_num = 0
    count = 0

    for data in loader:
        data1 = data[0].to(device)
        optimizer.zero_grad()
        x, edge_index, batch = data1.x, data1.edge_index, data1.batch
        output1 = model(x, edge_index, batch)

        data2 = data[1].to(device)
        optimizer.zero_grad()
        x, edge_index, batch = data2.x, data2.edge_index, data2.batch
        output2 = model(x, edge_index, batch)

        loss, distance = criterion(output1,output2,int(data1.y[0]!=data2.y[0]))

        count += 1
        if distance <= 0.4:
            num += 1

        if count % len(modelSet) == 0:
            if num >= level and data1.y[0].item() == 1:
                tp += 1
                correct += 1
            elif num >= level and data1.y[0].item() == 0:
                fp += 1
            elif num < level and data1.y[0].item() == 1:
                fn += 1
            elif num < level and data1.y[0].item() == 0:
                tn += 1 
                correct += 1
            total_num += num
            count = 0
            num = 0
    # print("pred_good: %d, pred_bad: %d" % (pred_good, pred_bad))
    return correct * len(modelSet) / len(loader.dataset), tp, fp, tn, fn, total_num / len(loader.dataset)

trainLoss = np.zeros(200)
testAcc = np.zeros(200)

tp = 0
fp = 0
tn = 0
fn = 0

level = 40

for epoch in range(1, 200):
    loss = train(epoch)
    acc, tp, fp, tn, fn, avg_num = myTest(sadataloader, level)
    print('Epoch {:3d},loss {:.5f}, tp: {:04d}, fp: {:04d}, tn: {:04d}, fn: {:04d}, acc {:.5f}, avg: {:03f}'.format(epoch,loss, tp, fp, tn, fn, acc, avg_num))
