import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


# 定义一个预热学习率调度器
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.total_epochs:
            factor = (self.last_epoch + 1) / self.total_epochs
        else:
            factor = 1.0
        return [base_lr * factor for base_lr in self.base_lrs]

    # 示例模型、损失函数和优化器


model = nn.Linear(10, 2)  # 简单的线性模型
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降优化器

# 设置预热epoch数
warmup_epochs = 5
# 总训练epoch数
total_epochs = 100

# 创建预热学习率调度器
warmup_scheduler = WarmUpLR(optimizer, total_epochs=warmup_epochs)

# 创建余弦退火学习率调度器
# 注意：这里我们不会立即使用它，而是会在预热结束后切换过去
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

# 模拟训练过程
for epoch in range(total_epochs):


    # 为了模拟训练，我们创建一些随机数据并计算损失
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 2)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # 在每个epoch开始时更新学习率
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        # 在预热结束后，切换到余弦退火调度器
        # 注意：由于CosineAnnealingLR期望从第0个epoch开始计数，
        # 因此我们需要调整其last_epoch属性以匹配当前的epoch数（相对于余弦退火周期）
        cosine_scheduler.last_epoch = epoch - warmup_epochs
        cosine_scheduler.step()
        # 由于我们手动设置了last_epoch，因此不需要再次调用optimizer.step()来更新参数
        # 但我们仍然需要更新学习率，这就是为什么我们调用cosine_scheduler.step()的原因
        # （尽管它通常与optimizer.step()一起使用来更新参数和学习率，但在这里我们只关心学习率更新）

    # 打印当前epoch、损失和学习率
    print(
        f'Epoch [{epoch + 1}/{total_epochs}], Loss: {loss.item():.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')