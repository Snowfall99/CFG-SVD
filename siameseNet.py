import torch.nn as nn
import torch
from torch import optim
import train4
from torch_geometric.nn import MessageingPassing, TopKPooling
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv = train4.GCNConv(256, 256)
        self.pool = TopKPooling(256, ratio=0.8)

    def forward_once(self, input):
        x, edge_index, batch = input.x, input.edge_index, input.batch
        x = self.conv(x, edge_index)
        x, edge_index, _, batch = self.pool(x, edge_index, None, batch)
        x = self.conv(x, edge_index)
        x, edge_index, _, batch = self.pool(x, edge_index, None, batch)
        x = self.conv(x, edge_index)
        x, edge_index, _, batch = self.pool(x, edge_index, None, batch)
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


net = SiameseNet()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0
train_number_epoches = 20

# train