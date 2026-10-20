import torch
from datetime import datetime
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 准备数据
demo_dir = Path(__file__).resolve().parent
data_dir = demo_dir / "data"
save_dir = demo_dir / "database"
save_dir.mkdir(parents=True, exist_ok=True)
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.01
train_set = datasets.MNIST(
    root=data_dir,
    train=True,# 训练集
    download=True,# 下载数据集
    transform=transforms.ToTensor() # 将图像转换为张量
)
test_set = datasets.MNIST(
    root=data_dir,
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
model = nn.Sequential(
    nn.Flatten(),          # 28×28 -> 784
    nn.Linear(784, 128),   # 784 -> 128
    nn.ReLU(), # 激活函数 ReLU(x)=max(0,x)
    nn.Linear(128, 10)     # 128 -> 10个数字类别
)
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
        "loss": avg_loss,
        "accuracy": accuracy
    })
    print(f"测试准确率：{accuracy * 100:.2f}%")

time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

df = pd.DataFrame(history)

df.to_csv(
    save_dir / f"mnist_lr{LR}_batch{BATCH_SIZE}_{time_str}.csv",
    index=False
)
