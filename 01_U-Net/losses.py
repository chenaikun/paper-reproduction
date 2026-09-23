import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss 专门衡量预测区域与真实区域的重叠程度（类似集合的交并比 IoU）。
    它对边界形态非常敏感，能有效防止预测结果糊成一片
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. logits 是模型最后一层未激活的输出，形状: [Batch, 2, H, W]
        # 对通道维度做 Softmax，取出属于“细胞类别（通道 1）”的概率图
        probs = F.softmax(logits, dim=1)[:, 1, :, :]  # 形状: [Batch, H, W]

        # 2. targets 是真实标签 (0 或 1)，转为浮点型
        targets = targets.float()

        # 3. 计算预测与真实的交集 (Intersection)
        intersection = (probs * targets).sum(dim=(1, 2))
        
        # 4. 计算并集总和
        total = (probs + targets).sum(dim=(1, 2))

        # 5. 计算 Dice 系数: (2 * 交集 + 平滑项) / (总和 + 平滑项)
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)

        # 损失值 = 1 - Dice 系数（Dice 越高，Loss 越低）
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    """
    将交叉熵损失与 Dice 损失结合：
    - CrossEntropy 负责逐个像素的分类准确率
    - DiceLoss 负责整张图的轮廓形状与边缘契合度
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)