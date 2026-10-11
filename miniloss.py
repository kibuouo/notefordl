import torch
import torch.nn as nn

x = torch.tensor(
    [[1.0], 
    [2.0],
    [3.0],
    [4.0]]
)
y = torch.tensor(
    [[3.0],
    [5.0],
    [7.0],
    [9.0]]
)
#y=wx+b
model=  nn.Linear(1,1) 
criterion = nn.MSELoss() # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)# 优化器 学习率0.01 
for epoch in range(200):
    # 前向传播
    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()  # 梯度清零
    loss.backward()        # 反向传播
    optimizer.step()       # 更新参数
    if (epoch+1) % 20 == 0:
        w=model.weight.item()
        b=model.bias.item()
        print(
            f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}, w: {w:.4f}, b: {b:.4f}"
        )