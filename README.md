# Model
基于控制流图的漏洞代码检测模型
# Content
- ```path2data.py```输入样本所在路径，生成对应的dataset
- ```train4.py``` 模型及主要训练函数
- ```word2vec.model```全局w2v模型
- ```Digraph.gv.pdf```torchviz自动生成的模型结构图
- ```sa_test4.py```基于GCN的相似性检测（模板匹配）程序
# Reference
[**Top-K Pooling**](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.TopKPooling) from Gao and Ji: [Graph U-Nets](https://arxiv.org/abs/1905.05178)(ICML 2019), Cangea et al: [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287)(NeurIPS-W 2018) and Knyazev et al.: [Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850) (ICLR-W 2019)  
**Source Code:**[[**Example**](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/proteins_topk_pool.py)]
