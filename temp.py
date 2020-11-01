import torch
from torch import nn
from torchviz import make_dot
from torch.autograd import Variable
from train4 import Net
import path2data1
from torch_geometric.data import DataLoader

datapath = r"/home/chenzx/train/tmp"

dataset = path2data1.loadPath2DataSet(datapath)
dataloader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)
model = Net()

torch.save(model.state_dict(), 'gcn5net.pth')

#for data in dataloader:
#    x, edge_index, batch = data.x, data.edge_index, data.batch
#    vis_graph = make_dot(model(x, edge_index, batch), params=dict(model.named_parameters()))
#    vis_graph.view()
