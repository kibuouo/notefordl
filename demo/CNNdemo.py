import torch
from datetime import datetime
import pandas as pd
import os
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0. 卷积层
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv(x)        # [64, 8, 28, 28]
        x = torch.relu(x)      # 激活函数
        x = self.pool(x)       # [64, 8, 14, 14]
        x = torch.flatten(x, 1) # [64, 1568]
        x = self.fc(x)         # [64, 10]
        return x


# 1. 准备数据
save_dir = Path("database")
save_dir.mkdir(parents=True, exist_ok=True)
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.01
train_set = datasets.MNIST(
    root="./data",
    train=True,# 训练集
    download=True,# 下载数据集
    transform=transforms.ToTensor() # 将图像转换为张量
)
test_set = datasets.MNIST(
    root="./data",
    train=False,# 测试集
    download=True,# 下载数据集
    transform=transforms.ToTensor() # 将图像转换为张量
)
loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False # 测试集不需要打乱
)
# 2. 定义模型
model = SimpleCNN()
model = model.to(device)
model = model.to(device)
# 3. 损失函数
criterion = nn.CrossEntropyLoss()

# 4. 优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR
)

# 5. 训练
history = []
for epoch in range(EPOCHS):
    model.train()  # 设置模型为训练模式
    total_loss = 0 

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: 平均损失 = {avg_loss:.4f}")
    # 训练结束后测试
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "test_accuracy": accuracy
    })
    print(f"测试准确率：{accuracy * 100:.2f}%")
os.makedirs("database", exist_ok=True)

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

df = pd.DataFrame(history)
# 保存训练指标
df.to_csv(
    save_dir / f"{time_str}_cnn_lr{LR}_batch{BATCH_SIZE}.csv",
    index=False,
    float_format="%.4f" # 小数位
)

# 保存训练好的模型参数
torch.save(
    model.state_dict(),
    save_dir / f"{time_str}_cnn_model.pth"
)