import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.data import HeteroData

class HGAT(torch.nn.Module):
    def __init__(self, in_channels_dict, out_channels, num_heads=4):
        super(HGAT, self).__init__()
        # 初始化异构图的卷积层
        # in_channels_dict 表示不同类型节点的输入维度，out_channels 是每个节点类型的输出维度
        self.conv1 = HeteroConv({
            'user': GATConv(in_channels_dict['user'], out_channels, heads=num_heads),
            'post': GATConv(in_channels_dict['post'], out_channels, heads=num_heads)
        }, aggr='sum')

        self.conv2 = HeteroConv({
            'user': GATConv(out_channels * num_heads, out_channels, heads=num_heads),
            'post': GATConv(out_channels * num_heads, out_channels, heads=num_heads)
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # 第1层GAT
        x_dict = self.conv1(x_dict, edge_index_dict)
        # 第2层GAT
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict
