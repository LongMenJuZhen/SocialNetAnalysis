import torch
from torch_geometric.data import HeteroData

# 创建异构图数据
data = HeteroData()

# 董监高节点特征
data['figure'].x = torch.randn(10, 16)  # 假设有10个用户节点，每个节点有16维特征
# 帖子节点特征
data['post'].x = torch.randn(5, 16)  # 假设有5个帖子节点，每个节点有16维特征

# user -> post 边
data['user', 'writes', 'post'].edge_index = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
# post -> user 边
data['post', 'written_by', 'user'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)

# 构建输入字典
x_dict = {
    'user': data['user'].x,
    'post': data['post'].x
}

edge_index_dict = {
    'user', 'writes', 'post': data['user', 'writes', 'post'].edge_index,
    'post', 'written_by', 'user': data['post', 'written_by', 'user'].edge_index
}
