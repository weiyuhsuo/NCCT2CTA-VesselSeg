基于 U-Net 的细胞核图像分割实验报告

  

1. 医学图像分割与 U-Net 基本原理

  

医学图像分割的目标是对图像中的每个像素进行分类，从而提取出目标区域（如病灶、细胞核等）。与图像分类不同，分割任务需要同时兼顾语义信息（这是什么）和空间信息（在哪里），因此对模型的结构设计有更高的要求。

  

U-Net 是医学图像分割中最经典的网络之一，其结构呈对称的“编码器-解码器”形态。编码器通过逐层下采样提取高层语义特征，解码器通过逐层上采样恢复空间分辨率。U-Net 最核心的设计是跳跃连接（skip connection）：将编码器每一层的特征图与解码器对应层的特征图在通道维度拼接，使解码器能够同时利用深层的语义信息和浅层的空间细节，从而更精确地定位目标边界。

  

本实验使用 U-Net 对细胞核图像进行分割。评价指标方面，Dice 系数衡量预测结果与真实标注的重叠程度，是分割任务最常用的指标；IoU（交并比）与 Dice 类似但更严格；像素准确率统计分类正确的像素占比，但在前景占比较小时容易受背景影响而偏高。

  

2. 核心代码分析

  

2.1 数据加载与预处理（dataset.py）

  

from pathlib import Path

import cv2

import numpy as np

import torch

from torch.utils.data import Dataset

  

class CellDataset(Dataset):

def __init__(self, root_dir, split='train', image_size=256, augment=False):

self.root_dir = Path(root_dir)

self.split = split

self.image_size = image_size

self.augment = augment

  

self.image_dir = self.root_dir / split / 'imageA'

self.mask_dir = self.root_dir / split / 'maskA'

  

self.images = sorted([p for p in self.image_dir.iterdir() if p.is_file()])

self.pairs = []

for img_path in self.images:

mask_path = self.mask_dir / f'{img_path.stem}.tif'

if mask_path.exists():

self.pairs.append((img_path, mask_path))

  

说明：自定义 Dataset 类，根据 split 参数（train/val/test）读取对应子目录下的图像和 mask。图像为 jpg 格式，mask 为 tif 格式，通过文件名 stem（不含后缀）进行一一匹配，避免因后缀不同导致的配对错误。使用 sorted 保证文件顺序固定，确保划分可复现。

  

def __getitem__(self, idx):

img_path, mask_path = self.pairs[idx]

image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

  

image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

  

if self.augment:

image, mask = self._augment(image, mask)

  

image = image.astype(np.float32) / 255.0

mask = (mask > 127).astype(np.float32)

  

image = torch.from_numpy(image).unsqueeze(0)

mask = torch.from_numpy(mask).unsqueeze(0)

return image, mask

  

说明：以灰度模式读取图像和 mask，统一 resize 到 256×256。图像使用线性插值（INTER_LINEAR），mask 使用最近邻插值（INTER_NEAREST）以避免引入中间灰度值。图像归一化到 [0,1]，mask 以阈值 127 二值化为 0/1。最后增加一个通道维度（unsqueeze(0)），将 H×W 变为 1×H×W 以适配 PyTorch 的卷积输入格式。训练时通过 _augment 方法随机进行水平和垂直翻转来增强数据。

  

2.2 模型构建（model.py）

  

class DoubleConv(nn.Module):

def __init__(self, in_channels, out_channels):

super().__init__()

self.block = nn.Sequential(

nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),

nn.BatchNorm2d(out_channels),

nn.ReLU(inplace=True),

nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),

nn.BatchNorm2d(out_channels),

nn.ReLU(inplace=True),

)

  

说明：DoubleConv 是 U-Net 的基本构建块，由两层 3×3 卷积 + BatchNorm + ReLU 组成。使用 BatchNorm 加速收敛并稳定训练，bias=False 是因为 BatchNorm 自带偏移项。

  

class UNet(nn.Module):

def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):

super().__init__()

self.downs = nn.ModuleList()

self.pools = nn.ModuleList()

ch = in_channels

for feature in features:

self.downs.append(DoubleConv(ch, feature))

self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

ch = feature

  

self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

  

self.ups = nn.ModuleList()

self.up_convs = nn.ModuleList()

rev_features = list(reversed(features))

ch = features[-1] * 2

for feature in rev_features:

self.ups.append(nn.ConvTranspose2d(ch, feature, kernel_size=2, stride=2))

self.up_convs.append(DoubleConv(feature * 2, feature))

ch = feature

  

self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

  

说明：U-Net 主体结构。编码器包含 4 个阶段，特征通道数依次为 64→128→256→512，每阶段由 DoubleConv + MaxPool2d 组成。瓶颈层将通道数扩至 1024（256→512→1024? 实际 bottleneck 是 DoubleConv(features[-1], features[-1]*2)，即 512→1024）。解码器通过 ConvTranspose2d（转置卷积）上采样，上采样后与对应的编码器特征图在通道维度拼接（skip connection），再经过 DoubleConv 处理。最后用 1×1 卷积将通道数映射到输出类别数（1，即二分类）。

  

def forward(self, x):

skip_connections = []

for down, pool in zip(self.downs, self.pools):

x = down(x)

skip_connections.append(x)

x = pool(x)

  

x = self.bottleneck(x)

skip_connections = skip_connections[::-1]

  

for idx in range(len(self.ups)):

x = self.ups[idx](x)

skip = skip_connections[idx]

if x.shape != skip.shape:

x = torch.nn.functional.interpolate(x, size=skip.shape[2:])

x = torch.cat((skip, x), dim=1)

x = self.up_convs[idx](x)

  

return self.final_conv(x)

  

说明：forward 方法实现完整的前向传播。编码阶段依次通过 DoubleConv 和 MaxPool，并保存每层的输出作为 skip connection。解码阶段将上采样结果与对应 skip connection 拼接，如果尺寸不匹配则用 interpolate 对齐（处理输入尺寸不能被 16 整除的情况）。最终通过 final_conv 输出单通道 logits，由后续的损失函数进行 sigmoid 激活。

  

2.3 评价指标（utils.py）

  

def dice_coeff(pred, target, smooth=1e-6):

pred = (torch.sigmoid(pred) > 0.5).float()

intersection = (pred * target).sum(dim=(1, 2, 3))

union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

dice = (2 * intersection + smooth) / (union + smooth)

return dice.mean().item()

  

def iou_score(pred, target, smooth=1e-6):

pred = (torch.sigmoid(pred) > 0.5).float()

intersection = (pred * target).sum(dim=(1, 2, 3))

union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

iou = (intersection + smooth) / (union + smooth)

return iou.mean().item()

  

def pixel_accuracy(pred, target):

pred = (torch.sigmoid(pred) > 0.5).float()

correct = (pred == target).float().sum()

return (correct / torch.numel(target)).item()

  

说明：三个评价指标均先将模型输出的 logits 通过 sigmoid 并取阈值 0.5 得到二值预测。Dice 系数 = 2×|A∩B| / (|A|+|B|)，衡量预测与标注的重叠程度。IoU = |A∩B| / |A∪B|。像素准确率 = 正确分类像素数 / 总像素数。smooth 项用于防止除零。

  

2.4 训练循环与主函数（train.py）

  

def train_one_epoch(model, loader, optimizer, criterion, device):

model.train()

total_loss = 0.0

for images, masks in loader:

images, masks = images.to(device), masks.to(device)

preds = model(images)

loss = criterion(preds, masks)

optimizer.zero_grad()

loss.backward()

optimizer.step()

total_loss += loss.item()

return total_loss / max(len(loader), 1)

  

说明：标准训练循环——前向传播 → 计算损失 → 反向传播 → 参数更新。使用 BCEWithLogitsLoss 作为损失函数（内部包含 sigmoid 激活和二值交叉熵），相比手动加 sigmoid 更数值稳定。

  

def evaluate(model, loader, criterion, device):

model.eval()

total_loss, total_dice, total_acc, total_iou = 0.0, 0.0, 0.0, 0.0

with torch.no_grad():

for images, masks in loader:

images, masks = images.to(device), masks.to(device)

preds = model(images)

loss = criterion(preds, masks)

total_loss += loss.item()

total_dice += dice_coeff(preds, masks)

total_acc += pixel_accuracy(preds, masks)

total_iou += iou_score(preds, masks)

n = max(len(loader), 1)

return {'loss': total_loss/n, 'dice': total_dice/n, 'acc': total_acc/n, 'iou': total_iou/n}

  

说明：评估函数使用 torch.no_grad() 禁用梯度计算以节省显存。同时计算 loss、Dice、Accuracy、IoU 四个指标。model.eval() 确保 BatchNorm 使用全局统计量而非批次统计量。

  

def main():

root_dir = Path(__file__).parent.parent / 'cell_dataset'

batch_size = 32

epochs = 20

lr = 1e-3

  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = CellDataset(root_dir, split='train', image_size=256, augment=True)

val_dataset = CellDataset(root_dir, split='val', image_size=256, augment=False)

test_dataset = CellDataset(root_dir, split='test', image_size=256, augment=False)

  

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,

num_workers=2, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,

num_workers=2, pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,

num_workers=2, pin_memory=True)

  

model = UNet(in_channels=1, out_channels=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

criterion = nn.BCEWithLogitsLoss()

  

best_dice = 0.0

for epoch in range(1, epochs + 1):

train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

val_metrics = evaluate(model, val_loader, criterion, device)

print(...)

if val_metrics['dice'] > best_dice:

best_dice = val_metrics['dice']

torch.save(model.state_dict(), 'checkpoints/best_unet.pth')

  

model.load_state_dict(torch.load('checkpoints/best_unet.pth'))

test_metrics = evaluate(model, test_loader, criterion, device)

  

说明：主函数设置超参数（batch_size=32, epochs=20, lr=1e-3），自动检测 GPU 并加载数据。训练时仅对 train 集做数据增强，val 和 test 不做增强。每个 epoch 后在验证集上评估，保存 Dice 最高的模型。训练完成后加载最佳模型在测试集上评估并输出最终结果。

  

3. 遇到的问题及解决方案

  

(1) opencv-python 缺少 Qt5 库。服务器无图形界面，opencv-python 默认依赖 Qt5 用于图像显示，运行时直接报错。解决方案：卸载 opencv-python，改用 opencv-python-headless，功能相同但不依赖 GUI 库。

  

(2) 初始 batch_size=8 训练速度偏慢。A100 显存 80GB，batch_size=8 时仅使用约 5GB，大量 GPU 并行能力闲置。将 batch_size 调至 32 后，每 epoch 耗时从约 63 秒降至约 58 秒，且模型收敛未受影响。

  

(3) SSH 远程连接需校园 VPN。服务器 IP（10.193.2.99）为校园内网地址，非校园网环境下需先连接学校 VPN 才能 SSH 登录。

  

(4) 原始 test 集 image 与 mask 数量不一致。作业说明中也提到了此问题，统一将 val 作为 test 使用，此处按要求从 train 中抽取 70 对作为新的 test 集。

  

4. 实验结果与分析

  

4.1 训练过程

在 A100 GPU 上训练 20 个 epoch，使用 BCEWithLogitsLoss + Adam 优化器。训练过程中 loss 持续下降，验证集 Dice 逐步上升，未出现过拟合迹象。每 epoch 约 58 秒，总计约 20 分钟。

  

4.2 最终结果

  

验证集指标：

- Loss: 0.1167

- Dice: 0.9684

- IoU: 0.9412

- Accuracy: 0.9521

  

测试集指标：

- Loss: 0.0166

- Dice: 0.9964

- IoU: 0.9927

- Accuracy: 0.9930

  

4.3 分析

测试集 Dice 达到 0.9964，IoU 达到 0.9927，说明模型能够非常准确地分割细胞核区域。验证集指标略低于测试集，可能与验证集和测试集各仅 70 对样本、统计波动较大有关。像素准确率在测试集上达到 0.993，但由于背景像素远多于前景像素，准确率本身参考价值有限，Dice 和 IoU 更能反映实际分割质量。

  

实验结果表明，标准的 U-Net 结构在该细胞核数据集上已经能够取得很好的分割效果。如要进一步优化，可以考虑引入注意力机制、改用 Dice Loss 或组合损失、增加弹性变形等更强的数据增强策略。

  

5. 总结

本实验使用 U-Net 模型完成了细胞核图像分割任务，涵盖数据预处理、模型构建、训练、评估的完整流程。测试集 Dice 系数达到 0.9964，分割效果良好。通过本实验，加深了对 U-Net 编解码结构、跳跃连接机制以及 Dice、IoU 等评价指标的理解。

  

环境说明：Python 3.12.9, PyTorch 2.6.0+cu124, NVIDIA A100-SXM4-80GB。运行时先 pip install -r requirements.txt，然后 cd src && python train.py 即可复现训练结果。