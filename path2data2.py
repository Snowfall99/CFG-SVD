import re, os
import utils2
import numpy as np
import gensim
import torch
import pydot
from torch_geometric.data import Data


w2vmodel = gensim.models.Word2Vec.load(r"/home/chenzx/w2vmodel/CPG190.model")

def Dot_to_Data(dot, label):
    """
    converts a CPG dot object to a class: 'torch_geometric.data.Data' instance
    """
    node_list = dot[0].get_nodes()
    if len(node_list) < 2:
        return -1
    edges = dot[0].get_edges()
    edge_i = []
    for edge in edges:
        src = edge.get_source()
        dst = edge.get_destination()
        for node in node_list:
            if node.get_name() == src:
                src_id = node_list.index(node)
            if node.get_name() == dst:
                dst_id = node_list.index(node)
        edge_i.append([src_id, dst_id])
    edge_index = torch.tensor(edge_i).t().contiguous()

    data = {}
    for i, node in enumerate(node_list):
        value = node.obj_dict['attributes']['label']
        tokens = utils2.preprocessing(value)
        tokenvec = np.zeros((128), dtype=float)
        token_number = len(tokens)
        for token in tokens:
            if token in w2vmodel.wv:
                tokenvec += (w2vmodel.wv[token])
            else:
                pass
        tokenvec = tokenvec / token_number
        data["x"] = [tokenvec] if i == 0 else data["x"] + [tokenvec]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1).long()
    data = Data.from_dict(data)
    data.num_nodes = len(node_list)
    data.x = data.x.float()
    data.y = torch.tensor([label], dtype=torch.long)

    return data



def Load_Dot_File(dotfileDirPath: str):
    for parent, dirs, files in os.walk(dotfileDirPath):
        for file in files:
            file = os.path.join(parent, file)
            ty = os.path.splitext(file)[1]
            if ty == ".dot":
                try:
                    if os.path.basename(file).find("good") != -1:
                        label = 1
                        yield file, label
                    elif os.path.basename(file).find("bad") != -1:
                        label = 0
                        yield file, label
                except:
                    continue


def Load_Path_2_DataSet(path):
    dataset = []
    for f, label in Load_Dot_File(path):
        try:
            dot = pydot.graph_from_dot_file(f)
            data = Dot_to_Data(dot, label)
            if data != -1:
                dataset.append(data)
        except Exception as e:
            print(e)
            
    return dataset