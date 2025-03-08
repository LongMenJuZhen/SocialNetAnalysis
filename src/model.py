import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
from torch_geometric.transforms import ToUndirected

# 读取数据
print("开始运行")
def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr
class SequenceEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class GenresEncoder:
    # The 'GenreEncoder' splits the raw column strings by 'sep' and converts
    # individual elements to categorical labels.
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = {g for col in df.values for g in col.split(self.sep)}
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


class IdentityEncoder:
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)
print("读取董监高数据")
executives_x, executives_mapping = load_node_csv(
    'data/processed/董监高/TMT_FIGUREINFO_final.csv', index_col='PersonID',encoders={
        'Nationality':IdentityEncoder(),
        'Gender':IdentityEncoder(),
        'Age':IdentityEncoder(),
        'Degree':IdentityEncoder(),
        'Resume':SequenceEncoder(),
        'Funback':IdentityEncoder(),
        'OveseaBack':IdentityEncoder(),
        'Academic':IdentityEncoder(),
        'FinBack':IdentityEncoder()
    })
print("读取企业数据")
companies_x, companies_mapping = load_node_csv(
    'data/processed/企业/FI_T7_final.csv', index_col='Stkcd', encoders={
        'Accper':IdentityEncoder(),
        'F070301B':IdentityEncoder()
    })
print("读取职位数据")
edge_index, edge_label = load_edge_csv(
    'data/processed/职位/TMT_POSITION_final_numtime.csv',
    src_index_col='PersonID',
    src_mapping=executives_mapping,
    dst_index_col='Stkcd',
    dst_mapping=companies_mapping,
    encoders={'Reptdt': IdentityEncoder(dtype=torch.float),
    'PositionID':IdentityEncoder(),
    'StartDate':IdentityEncoder(),
    'ServiceStatus':IdentityEncoder(),
    'Tenure':IdentityEncoder(),
    'GTAPosition':IdentityEncoder(),
    'PaidSign':IdentityEncoder(),
    'TotalSalary':IdentityEncoder(),
    "SharEnd":IdentityEncoder(),
    'IsMTMT':IdentityEncoder(),
    'IsMTB':IdentityEncoder(),
    'IsIdirecotr':IdentityEncoder(),
    'IsDuality':IdentityEncoder(),
    'IsSupervisor':IdentityEncoder(),
    'ServicePosition':IdentityEncoder(),
    'IsCocurP':IdentityEncoder()},
)
# companies = pd.read_csv('data/processed/企业/FI_T7_final.csv')
# executives = pd.read_csv('data/processed/董监高/TMT_FIGUREINFO_final.csv')
# positions = pd.read_csv('data/processed/职位/TMT_POSITION_final.csv')

# 创建异构图数据
data = HeteroData()
data['executives'].num_nodes = len(executives_mapping)  # Users do not have any features.
data['companies'].x = companies_x
data['executives', 'positions', 'companies'].edge_index = edge_index
data['executives', 'positions', 'companies'].edge_label = edge_label
print(data)
# # 添加公司节点
# data['company'].x = torch.tensor(companies[['Stkcd']].values, dtype=torch.float)
# data['company'].y = torch.tensor(companies['F070301B'].values, dtype=torch.float)

# # 添加董监高节点
# data['executive'].x = torch.tensor(executives.drop(columns=['PersonID']).values, dtype=torch.float)

# # 添加任职关系边
# edge_index = torch.tensor([positions['PersonID'].values, positions['Stkcd'].values], dtype=torch.long)
# data['executive', 'works_at', 'company'].edge_index = edge_index

# # 转换为无向图
# data = ToUndirected()(data)

# # 定义HAN模型
# class HAN(nn.Module):
#     def __init__(self, in_channels_dict, out_channels, hidden_channels=128, heads=8):
#         super().__init__()
#         self.han_conv = HANConv(in_channels_dict, hidden_channels, heads=heads, dropout=0.6, metadata=data.metadata())
#         self.lin = nn.Linear(hidden_channels, out_channels)

#     def forward(self, x_dict, edge_index_dict):
#         out = self.han_conv(x_dict, edge_index_dict)
#         out = self.lin(out['company'])
#         return out

# # 初始化模型
# in_channels_dict = {
#     'company': data['company'].x.size(1),
#     'executive': data['executive'].x.size(1)
# }
# model = HAN(in_channels=in_channels_dict, out_channels=1)

# # 选择设备
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
#     device = torch.device('cpu')
# data, model = data.to(device), model.to(device)

# # 初始化懒惰模块
# with torch.no_grad():
#     out = model(data.x_dict, data.edge_index_dict)

# # 定义优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

# # 训练函数
# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x_dict, data.edge_index_dict).squeeze()
#     loss = F.mse_loss(out, data['company'].y)
#     loss.backward()
#     optimizer.step()
#     return float(loss)

# # 测试函数
# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(data.x_dict, data.edge_index_dict).squeeze()
#     loss = F.mse_loss(pred, data['company'].y)
#     return float(loss)

# # 训练和验证
# for epoch in range(1, 201):
#     loss = train()
#     test_loss = test()
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}')