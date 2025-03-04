# 初始化模型
model = HGAT(in_channels_dict={'user': 16, 'post': 16}, out_channels=8, num_heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 假设每个用户节点和每个帖子节点都有标签
labels = torch.randint(0, 2, (10,))  # 用户的标签
labels_post = torch.randint(0, 2, (5,))  # 帖子的标签

# 训练过程
model.train()
for epoch in range(200):
    optimizer.zero_grad()

    # 前向传播
    out = model(x_dict, edge_index_dict)

    # 假设我们做的是节点分类任务
    loss_user = criterion(out['user'], labels)  # 用户节点的损失
    loss_post = criterion(out['post'], labels_post)  # 帖子节点的损失

    # 总损失
    loss = loss_user + loss_post
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
