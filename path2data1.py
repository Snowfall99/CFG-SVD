import re, os
import utils
import gensim
import torch
import pydot
import numpy as np
from torch_geometric.data import Data


pattern_CWE = re.compile(r'CWE[1-9]\d*')
w2vmodel = gensim.models.Word2Vec.load(r"/home/chenzx/train/word2vec.model")



def Dot_to_Data(dot, cwe_label, label):
    """
    Converts an obj:'pydot.dot' to a class: 'torch_geometric.data.Data' instance

    :param dot: pydot graph
    :param cwe_label: CWE***
    :param label: 1-good, 0-bad
    :return: torch_geometric.data.Data
    """
    node_list = dot[0].get_nodes()
    if len(node_list) < 2:
        return -1
    edges = dot[0].get_edges()
    edge_i = []
    for edge in edges:
        src = edge.get_source().split(":")[0]
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
        tokens = utils.preprocessing(value)
        tokenvec = np.zeros((256,), dtype=float)
        token_number = len(tokens)
        for token in tokens:
            if token in w2vmodel.wv:
                tokenvec += (w2vmodel.wv[token])
            else:
                pass
        tokenvec = tokenvec / token_number
        data["x"] = [tokenvec] if i ==0 else data["x"] + [tokenvec]

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
    data.__setitem__('cwe', cwe_label)

    return data


def loadDotFile(dotfileDirPath: str):
    for parent, dirs, files in os.walk(dotfileDirPath):
        for file in files:
            file = os.path.join(parent, file)
            ty = os.path.splitext(file)[1]
            if ty == ".dot":
                try:
                    cwe_label = str(pattern_CWE.search(file).group(0))
                    if os.path.basename(file).find("good") != -1:
                        label = 1
                        yield file, cwe_label, label
                    elif os.path.basename(file).find("bad") != -1:
                        label = 0
                        yield file, cwe_label, label
                except:
                    print("error", file)
                    continue


def loadPath2DataSet(path):
    dataset = []
    for f, cwe_label, label in loadDotFile(path):
        try:
            dot = pydot.graph_from_dot_file(f)
            data = Dot_to_Data(dot, cwe_label, label)
            if data != -1:
                dataset.append(data)
        except Exception as e:
            # print(e)
            pass

    return dataset