# 简介

湖南大学 社会网络分析 的 大作业

利用图注意力网络~~水一个~~对上市公司高管的社会网络进行研究

## 关于数据库

版权限制，并不会上传。

# 环境配置

我们必须假设你的电脑符合以下的条件：

* 网络状况良好
* 是windows操作系统
* 有英伟达显卡
* 未安装python及相关工具链

~~什么真空中的球形鸡~~

那么通过以下命令可以配置环境（以下命令记得用管理员权限执行）

```
#安装python312
winget install --scope machine python.python.3.12
#安装poetry，一个python的npm like的管理工具
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
#进入项目文件夹
cd <PATH TO THIS PROJECT>
#安装依赖包
poetry sync
```

如果你已经安装了其他版本的python

可以通过 `poetry env use "C:\Program Files\Python312\python.exe" `的方式指定python解释器。

## 关于拓展特别是torch是怎么安装的

理想情况下，不必再执行以下命令，这些命令记录了toml文件是怎么由poetry生成的

```
poetry source add --priority=supplemental pytorch_cu124 https://download.pytorch.org/whl/cu124
poetry source add --priority=supplemental pyg_cu124 https://data.pyg.org/whl/torch-2.5.0+cu124.html
poetry add --source pytorch_cu124 torch
poetry add torch_geometric
poetry add --source pyg_cu124 pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
```
