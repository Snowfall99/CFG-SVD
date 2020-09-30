import path2data1
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

startTime = time.time()

good_label = 0
bad_label = 0

datapath = r"/home/chenzx/testcases/CWE762_Mismatched_Memory_Management_Routines"

dataset = path2data1.loadPath2DataSet(datapath)

random.shuffle(dataset)
lenDataset = len(dataset)
lenTrainset = int(0.7*lenDataset)
lenValidset = int(0.2*lenDataset)
lenTestset = lenDataset - lenTrainset - lenValidset
print("数据集总量：%d 训练集：%d 验证集：%d 测试集：%d" % (lenDataset, lenTrainset, lenValidset, lenTestset))
trainSet = dataset[:lenTrainset]

# 验证集
validateSet = dataset[lenTrainset:lenTrainset+lenValidset]
# 测试集
testSet = dataset[lenTrainset+lenValidset:]
# 加载训练集
trainloader = DataLoader(dataset=trainSet,batch_size=32,shuffle=True)
testloader = DataLoader(dataset=testSet, batch_size=32, shuffle=True)
for test_data in testSet:
    if test_data["y"] == 1:
        good_label += 1
    elif test_data["y"] == 0:
        bad_label += 1;
print("good_label: %d bad_label: %d" % (good_label, bad_label))

finishDataLoadingTime = time.time()

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
        # self.conv1 = GraphConv(128, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GCNConv(128, 128)
        # self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GCNConv(128, 128)
        # self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.conv4 = GCNConv(128, 128)
        self.pool4 = TopKPooling(128, ratio=0.8)
        self.conv5 = GCNConv(128, 128)
        self.pool5 = TopKPooling(128, ratio=0.8)

        self.convAtt1 = torch.nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, stride=2)
        self.poolAtt1 = torch.nn.MaxPool1d(kernel_size=1, stride=2)
        self.convAtt2 = torch.nn.Conv1d(64, 16, kernel_size=1, stride=2)
        self.poolAtt2 = torch.nn.MaxPool1d(kernel_size=1, stride=2)
        self.convAtt3 = torch.nn.Conv1d(16, 2, kernel_size=1, stride=2)
        self.poolAtt3 = torch.nn.MaxPool1d(kernel_size=1, stride=2)

        self.convAtt4 = torch.nn.Conv1d(2, 16, kernel_size=1, stride=2)
        self.poolAtt4 = torch.nn.MaxPool1d(kernel_size=1, stride=2)
        self.convAtt5 = torch.nn.Conv1d(16, 64, kernel_size=1, stride=2)
        self.poolAtt5 = torch.nn.MaxPool1d(kernel_size=1, stride=2)
        self.convAtt6 = torch.nn.Conv1d(64, 256, kernel_size=1, stride=2)
        self.poolAtt6 = torch.nn.MaxPool1d(kernel_size=1, stride=2)


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




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train(epoch):
    model.train()

    loss_all = 0
    for data in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        x, edge_index, batch = data.x, data.edge_index, data.batch
        output = model(x, edge_index, batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(trainSet)

def myTest(loader):
    model.eval()
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    pred_good = 0
    pred_bad = 0
    for data in loader:
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        preds = model(x, edge_index, batch).max(dim=1)[1]
        correct += preds.eq(data.y).sum().item()
        for i in range(len(preds)):
            if preds[i] == 1 and data.y[i] == 1:
                tp += 1
                pred_good += 1
            elif preds[i] == 1 and data.y[i] == 0:
                fp += 1
                pred_good += 1
            elif preds[i] == 0 and data.y[i] == 0:
                tn += 1
                pred_bad += 1
            elif preds[i] == 0 and data.y[i] == 1:
                fn += 1
                pred_bad += 1
    # print("pred_good: %d, pred_bad: %d" % (pred_good, pred_bad))
    return correct / len(loader.dataset), tp, fp, tn, fn, pred_good, pred_bad

for epoch in range(1, 201):
    loss = train(epoch)
    train_acc, tp, fp, tn, fn, pred_good, pred_bad = myTest(trainloader)
    test_acc, tp, fp, tn, fn, pred_good, pred_bad = myTest(testloader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f},Test Acc: {:.5f}, TP: {:04d}, FP: {:04d}, TN: {:04d}, FN: {:04d}, Pred_good: {:04d}, Pred_bad: {:04d}'.
          format(epoch, loss, train_acc, test_acc, tp, fp, tn, fn, pred_good, pred_bad))

finishTime = time.time()

print("dataloading time: ", finishDataLoadingTime-startTime)
print("training and testing time: ", finishTime-finishDataLoadingTime)