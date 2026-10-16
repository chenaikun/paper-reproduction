# GFL 单层损失模拟：逐行中文讲解版
# 本文件只添加注释，执行代码与 debug_loss_single.py 一致，原文件不修改。
# 在本文件夹运行：python debug_loss_single_annotated.py --trace
# 调试：在 run_debug 第一行设断点，F5启动，F10单步，F11进入函数。
#
# 阅读顺序：构造张量 -> 展平 -> 选择正样本 -> Integral -> 框解码 -> IoU
# -> GIoU / DFL / QFL -> 加权总损失 -> backward -> optimizer.step。
# N=2张图，每张16个位置，共32个位置；P=3个正样本；80个类别。
# 每个位置回归4条边，每条边17个logits，共68维。
#
# 汇报必须说明的简化：
# 1. 本例没有图片、backbone/FPN和ATSS，预测与正样本标签是人工构造。
# 2. anchors=[x,y,x,y]只是取参考点用的零面积占位框，不是真实anchor尺寸。
# 3. 本例bbox_targets是特征图单位的l/t/r/b；正式GFLHead入口为像素xyxy，
#    再除stride并encode；同名变量在这两种入口中的含义不同。
# 4. 回归损失展示单层局部值；正式loss_by_feat还会在汇总各层后
#    除以回归权重总和，分布式训练还涉及跨进程归一化。
# 5. 本例最后才乘loss_weight；正式损失模块已乘系数，不能重复乘。
# 6. 本例SGD直接优化模拟logits，真实训练SGD优化model.parameters()。
# 7. backward计算梯度，step更新数值；旧概率、框和loss不会自动重算。
# 8. 概率分布能表达定位的不确定程度，但不自动保证不确定性经过校准。
# 9. 正样本身份由监督分配决定，不能因分类分数低就说它不是真实目标。
# 10. 本例只演示单次更新；重复调用run_debug会复用已更新的全局张量。
#
# 建议不要进入PyTorch底层，先对照每个自定义函数的输入、输出、形状。

# 导入命令行参数工具；后面用它识别 --trace 和 --pdb，控制是否展示中间值、是否暂停。
import argparse
# 导入 Python 系统模块；这里修改 sys.path，让 Python 能找到本地 MMDetection。
import sys
# 导入路径工具 Path；用 / 拼接路径，可以避免手写 Windows 反斜杠。
from pathlib import Path
# 导入 PyTorch：负责张量、自动求导和优化器。
import torch
# 把 PyTorch 的函数接口简称为 F，后面通过 F 调用 softmax 和交叉熵。
import torch.nn.functional as F

# 固定随机种子，便于汇报时重复运行并得到相同的随机预测值。
# 固定随机数生成器的起点；每次从头运行，随机 logits 相同，方便对照调试。
torch.manual_seed(0)

# ----------------- 1. 解决目录路径导入问题 -----------------
# 当前脚本位于项目根目录的“汇报”文件夹，需要向上一级寻找项目。
# __file__ 是当前脚本文件名；resolve() 得到绝对路径，parent 取得汇报文件夹。
CURRENT_DIR = Path(__file__).resolve().parent
# parent 是 2020_neurips_gfl 项目目录。
PROJECT_ROOT = CURRENT_DIR.parent

# 将根目录和 mmdetection 路径加入到 sys.path 中
# 检查项目路径是否已经在模块搜索列表里，避免重复插入。
if str(PROJECT_ROOT) not in sys.path:
    # 把项目路径放到搜索列表最前面；import 会优先从这里查找模块。
    sys.path.insert(0, str(PROJECT_ROOT))

# 拼接本地 MMDetection 仓库路径；这里的 / 是 Path 的路径运算符。
mmdet_path = PROJECT_ROOT / "mmdetection"
# 目录存在且尚未被加入搜索路径时，才执行下面的插入。
if mmdet_path.exists() and str(mmdet_path) not in sys.path:
    # 优先搜索本地 mmdetection 下的 mmdet 包；已经导入的模块不会因此重新加载。
    sys.path.insert(0, str(mmdet_path))

# 验证导入 (根据你的实际类名导入，如果是官方 gfl 可直接导入)
# 尝试导入 MMDetection；若发生下面指定的 ImportError，就进入 except 分支。
try:
    # 只验证 GFLHead 是否可导入；本脚本没有实例化它，下面的损失是手写张量模拟。
    from mmdet.models.dense_heads import GFLHead
    # 导入成功时显示提示；不表示已经运行了正式 GFLHead。
    print("成功导入 MMDetection 模块！")
# 只捕获导入错误，e 保存错误原因；其他类型异常不由此分支处理。
except ImportError as e:
    # 显示导入失败原因；主体只需要 torch，因此这种导入错误不会阻止纯张量演示。
    print(f"导入提示: {e}，将直接进行纯张量逻辑调试。")


#基础数据

# N 是 batch size：模拟一次处理 2 张图。
N=2 #batchsize为2
# 模拟 COCO 的 80 个前景类别，内部类别索引为 0 到 79。
classes_nums=80
# 只模拟一个 4×4 特征层，每张图有 16 个候选位置；不是原图只有 4×4 像素。
H,W=4,4 
# 每条边的离散支持点是 0、1、…、16，共 17 个；16 不是候选数。
reg_max=16  #最大离散值
# 每张图在这一层有 H×W=16 个位置；当前模拟每个位置对应一个预测。
anchors_num=H*W  #一张特征图全部特征点
# 特征图沿 x、y 方向的步长都为 8；特征图距离 1 相当于原图距离 8 像素。
stride=(8,8) #下采样倍数


#构造head预测值
# 生成形状 [2,80,4,4] 的随机分类 logits；尚未 sigmoid。requires_grad=True 使其成为可求导的叶子张量。
cls_score=torch.randn(N,classes_nums,H,W,requires_grad=True)

# 生成 [2,68,4,4] 的随机回归 logits；每个位置有 l/t/r/b 四组，每组 17 个值，可求导。
bbox_pred=torch.randn(N,4*(reg_max+1),H,W,requires_grad=True)

#构造几何基准
# 生成 4×4 网格索引；ij 表示第一维按行 y 排列、第二维按列 x 排列。
y_grid,x_grid=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
# 把列索引转换为像素参考点 x：先加半格，再乘 8，例如第 0 列得到 4 像素。
x_centers=(x_grid.float()+0.5)*stride[0]
# 把行索引转换为像素参考点 y；这里参考点是网格位置，不一定是真实目标框的中心。
y_centers=(y_grid.float()+0.5)*stride[1]
# 先把每张 4×4 网格展平为 16 项，再沿最后一维组成 [16,2]，每行为 [x,y]。
points=torch.stack([x_centers.flatten(),y_centers.flatten()],dim=-1)
#.unsqueeze(0):把这叠卡片装进 1 个文件夹里 ---> 得到 [1 个文件夹, 16 张卡片, 4 个坐标]
#.repeat(2, 1, 1):按 (2, 1, 1) 放到复印机里 ---> 得到 [2 个文件夹, 16 张卡片, 4 个坐标](正好对应 Batch 中的 2 张图片，每张图片各分到一个装有 16 张卡片的文件夹)
# 把 [x,y] 复制为 [x,y,x,y]，再复制到两张图，得到 [2,16,4]；这些是零面积占位框，只用于取参考点，不是真实 anchor 尺寸。
anchors=torch.cat([points,points],dim=-1).unsqueeze(0).repeat(N,1,1)

#构造监督信息
# 建立 [2,16] 的整数标签，先全部填 80；0…79 为前景，80 是背景标记，不是第 81 个输出通道。
labels=torch.full((N,anchors_num),classes_nums,dtype=torch.long)  #类别数设置为80  一般用于当作负样本或者背景


#加入几个正样本点
# 把第一张图索引 5 的位置人工设为类别 1 的正样本；Python 索引从 0 开始。
labels[0,5]=1   #第一张图的第6个点类别为1  
# 把第二张图索引 2 的位置人工设为类别 0 的正样本。
labels[1,2]=0   #第二张图的第3个点类别为0
# 再设第一张图索引 0 为类别 1；总共 3 个正样本。这里跳过了真实训练中的 ATSS 分配。
labels[0,0]=1   #第一张图的第1个点类别为1

# 建立 [2,16] 的分类位置权重，全部为 1，表示当前所有位置都参与分类损失。
label_weights = torch.ones((N,anchors_num),dtype=torch.float32)

# 正样本回归目标。这里直接使用特征图坐标单位，便于和中心点相减解码。
# 建立 [2,16,4] 的占位目标；本脚本填的是特征图单位的 [l,t,r,b]，正式 GFLHead 入口的 bbox_targets 则是像素 xyxy 框。
bbox_targets=torch.zeros((N,anchors_num,4),dtype=torch.float32)
# 给第一个指定正样本填四边距离；相对其参考点定义，已经除过 stride，不要再除一次。
bbox_targets[0,5]=torch.tensor([3.2, 5.1, 4.0, 2.5])
# 给第二张图的正样本填 [l,t,r,b]；四条边分别有一个连续 DFL 目标。
bbox_targets[1,2]=torch.tensor([8.6, 2.3, 7.4, 6.1])
# 给第一张图索引 0 填目标；当前合成距离都在 [0,16) 内，floor(y)+1 不会越过候选索引 16。
bbox_targets[0,0]=torch.tensor([4.5, 4.5, 3.0, 3.0])

#控制正样本数量的兜底
#防止正样本出现数据为0
# 统计标签小于 80 的位置，得到 3；下限 1 防止除零。当前无负标签；若加入 ignore=-1，还需同时判断 labels>=0。
avg_factor=max(int((labels<classes_nums).sum()),1)


# ----------------- 3. 用纯 PyTorch 模拟 GFL 的三个损失 -----------------
# 这些函数是论文公式和 GFLHead 中的调用顺序。



# 定义 Integral 解码函数；输入 [P,68]，P 是正样本数，当前 P=3。
def integral_decode(bbox_logits):
    # 函数说明字符串：说明用途及输入输出，不执行张量计算。
    """Integral: 4 组 (reg_max + 1) 个 logits -> 4 条连续边距离。"""
    # bbox_logits: [num_positive, 4 * 17]
    # [3,68] 变成 [12,17]；-1 自动推算为 3×4，每行对应一条边的 17 个 logits。
    distributions = bbox_logits.reshape(-1, reg_max + 1)
    # 沿每行的 17 个候选距离做 softmax，得到每条边总和为 1 的概率；这不是四条边之间竞争。
    probabilities = F.softmax(distributions, dim=1)
    # 准备创建固定距离坐标 [0,1,…,16]；它不是可训练网络权重。
    project = torch.arange(
        # arange 的右端不包含在内；设置与 logits 相同的设备和浮点类型，避免 CPU/GPU 或类型不匹配。
        reg_max + 1, device=bbox_logits.device, dtype=bbox_logits.dtype)
    # 按列广播距离向量，计算每行的 Σ p_i×i，得到 12 个连续距离；梯度可通过乘法、求和和 softmax 传回 logits。
    distances = (probabilities * project).sum(dim=1)
    # 把 12 个距离按四条边分组，返回 [3,4]，列顺序为 l、t、r、b。
    return distances.reshape(-1, 4)


# 定义距离解码：输入参考点 [P,2] 和距离 [P,4]；两者必须使用相同坐标单位。
def distances_to_boxes(centers, distances):
    # 函数说明字符串：说明用途及输入输出，不执行张量计算。
    """把 [left, top, right, bottom] 距离解码为 xyxy 框。"""
    # 沿列拆成四个 [P] 向量；unbind 会去掉被拆分的那一维。
    left, top, right, bottom = distances.unbind(dim=1)
    # 把参考点拆为 x、y 两个向量；变量名 cx/cy 不代表这个点一定是目标框几何中心。
    cx, cy = centers.unbind(dim=1)
    # 左上角向参考点左上方减距离、右下角加距离，组成 [P,4] 的 [x1,y1,x2,y2] 框。
    return torch.stack([cx - left, cy - top, cx + right, cy + bottom], dim=1)


# 定义一一对应的 IoU：第 i 个预测只比较第 i 个真值；不是生成 P×P 的两两比较矩阵。
def aligned_iou(boxes1, boxes2, eps=1e-7):
    # 函数说明字符串：说明用途及输入输出，不执行张量计算。
    """计算一一对应的 IoU；第 i 个 boxes1 只和第 i 个 boxes2 比较。"""
    # 交集左边界取两个框左边界较大的值，因为交集必须同时在两框内部。
    inter_left = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    # 交集上边界取较大的 y1。
    inter_top = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    # 交集右边界取较小的 x2。
    inter_right = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    # 交集下边界取较小的 y2。
    inter_bottom = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    # 交集宽度若为负，说明水平方向不相交；截为 0，避免负面积。
    inter_w = (inter_right - inter_left).clamp(min=0)
    # 同理计算非负交集高度。
    inter_h = (inter_bottom - inter_top).clamp(min=0)
    # 交集面积 = 交集宽×高，每个对应框对得到一个数。
    inter_area = inter_w * inter_h

    # 计算第一个框面积，先计算非负宽度，下一行乘非负高度。
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)  # 第一个框的高度；与前一行宽度相乘得到面积。
    # 计算第二个框面积，先计算非负宽度。
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)  # 第二个框的高度；与前一行宽度相乘得到面积。
    # IoU=交集/并集；并集=A+B-交集，eps 给分母下限以避免除零。
    return inter_area / (area1 + area2 - inter_area).clamp(min=eps)


# 定义一一对应的 GIoU 损失；输入是 xyxy 框，不是 logits 或概率分布。
def aligned_giou_loss(boxes1, boxes2, eps=1e-7):
    # 函数说明字符串：说明用途及输入输出，不执行张量计算。
    """计算一一对应的 GIoU Loss = 1 - GIoU。"""
    # 先算 IoU；此处预测框不 detach，后续 GIoU 可通过框解码和 Integral 影响回归 logits。
    iou = aligned_iou(boxes1, boxes2, eps)

    # 最小外接矩形的左边界取更小的 x1；与交集的取法相反。
    enc_left = torch.minimum(boxes1[:, 0], boxes2[:, 0])
    # 外接矩形的上边界取更小的 y1。
    enc_top = torch.minimum(boxes1[:, 1], boxes2[:, 1])
    # 外接矩形的右边界取更大的 x2。
    enc_right = torch.maximum(boxes1[:, 2], boxes2[:, 2])
    # 外接矩形的下边界取更大的 y2。
    enc_bottom = torch.maximum(boxes1[:, 3], boxes2[:, 3])
    # 外接矩形面积 C，先计算宽度。
    enc_area = (enc_right - enc_left).clamp(min=0) * \
        (enc_bottom - enc_top).clamp(min=0)  # 再乘外接矩形高度；C 是包住两个框的最小轴对齐矩形。

    # GIoU 中的 union_area 需要交集面积；这里重新计算交集，代码更直观。
    # 交集左边界取两个框左边界较大的值，因为交集必须同时在两框内部。
    inter_left = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    # 交集上边界取较大的 y1。
    inter_top = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    # 交集右边界取较小的 x2。
    inter_right = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    # 交集下边界取较小的 y2。
    inter_bottom = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    # 重新计算交集面积，先取非负交集宽度。
    inter_area = (inter_right - inter_left).clamp(min=0) * \
        (inter_bottom - inter_top).clamp(min=0)  # 再乘非负交集高度。
    # 计算第一个框面积，先计算非负宽度，下一行乘非负高度。
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)  # 第一个框的高度；与前一行宽度相乘得到面积。
    # 计算第二个框面积，先计算非负宽度。
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)  # 第二个框的高度；与前一行宽度相乘得到面积。
    # 并集面积 = 两个框面积之和减去重复计算的交集。
    union_area = area1 + area2 - inter_area
    # GIoU=IoU-(C-并集)/C；额外惩罚外接矩形中的空白区域，GIoU 可能为负。
    giou = iou - (enc_area - union_area) / enc_area.clamp(min=eps)
    # 相似度越大越好，损失越小越好；用 1-GIoU 转换目标，完全重合时损失为 0。
    return 1 - giou


# 定义 QFL；logits 为 [32,80]，labels/quality 为 [32]，beta=2 控制预测偏差的调制强度。
def quality_focal_loss(logits, labels, quality, beta=2.0):
    # 函数说明字符串：说明用途及输入输出，不执行张量计算。
    """QFL 的最小实现：负样本质量为0，正样本真实类别质量为 IoU。"""
    # 分类用逐类别 sigmoid；80 类概率不要求加起来等于 1，区别于回归每条边上的 softmax。
    probabilities = logits.sigmoid()
    # 建立 [32,80] 全零目标；用于背景位置以及正样本的非真实类别。
    zero_target = torch.zeros_like(logits)

    # 先把所有位置当成背景，和 mmdet 的 quality_focal_loss 第 35-40 行一致。
    # 计算数值稳定的二元交叉熵，直接输入 logits，不要先把 sigmoid 概率再次传给此函数。
    loss = F.binary_cross_entropy_with_logits(
        # 先按目标 0 得到每个位置每个类别的 BCE，再乘 p^beta；reduction='none' 保留 [32,80]。
        logits, zero_target, reduction='none') * probabilities.pow(beta)

    # 找前景位置：类别 0…79；80 为背景，不选中。
    positive = (labels >= 0) & (labels < classes_nums)
    # 布尔掩码变成正样本行号 [0,5,18]；squeeze(1) 把 [3,1] 变成 [3]。
    positive_indices = positive.nonzero(as_tuple=False).squeeze(1)
    # 存在正样本才覆盖相应类别损失；numel() 返回张量元素数。
    if positive_indices.numel() > 0:
        # 取各正样本真实类别，得到 [1,1,0]；每个行号和类别索引成对使用。
        positive_labels = labels[positive_indices]
        # 准备计算正样本真实类别的质量偏差 y-p。
        quality_gap = quality[positive_indices] - probabilities[
            # 成对索引取每个正样本真实类别的概率；不是取所有行列组合。
            positive_indices, positive_labels]
        # 计算正样本真实类别的 BCE；目标由 0 改为当前预测框的 IoU。
        positive_loss = F.binary_cross_entropy_with_logits(
            # 输入各正样本真实类别的原始 logits，形状 [3]。
            logits[positive_indices, positive_labels],
            # 对应的连续监督目标是 IoU，取值在 [0,1]；目标无需是整数。
            quality[positive_indices],
            # 乘 |y-p|^beta，得到 QFL；预测越接近质量标签，调制因子越小。
            reduction='none') * quality_gap.abs().pow(beta)
        # 只覆盖正样本真实类别这一格；正样本其他类别、所有背景类别仍保留目标 0 的损失。
        loss[positive_indices, positive_labels] = positive_loss

    # label_weights 和 avg_factor 对应 GFLHead 对 QFL 的加权与归一化。
    # 先沿 80 类求和为 [32]，乘位置权重后全局求和，再除 3 个正样本；因此随机 logits 的 QFL 可能较大。
    return (loss.sum(dim=1) * label_weights.reshape(-1)).sum() / avg_factor


# ----------------- 4. 调试输出和断点 -----------------
# 定义阶段输出工具；只改变显示，不改变损失计算。
def trace_print(enabled, title, message):
    # 函数说明字符串：说明用途及输入输出，不执行张量计算。
    """只在 --trace 模式打印中间量，避免普通运行时输出过多。"""
    # 当开关为 True 才执行缩进内的操作。
    if enabled:
        # f 字符串把标题和中间量填入输出，\n 表示换行。
        print(f"\n[{title}]\n{message}")


# 定义可选断点工具；--pdb 启用后会在阶段之间暂停。
def pause_if_requested(enabled, title):
    # 函数说明字符串：说明用途及输入输出，不执行张量计算。
    """在 --pdb 模式暂停，让你检查当前作用域中的局部变量。"""
    # 当开关为 True 才执行缩进内的操作。
    if enabled:
        # 先说明当前停在哪个阶段，方便定位计算进度。
        print(f"\n即将暂停：{title}")
        # 显示 pdb 命令；注意当前栈帧在辅助函数中，查看 run_debug 的局部变量要先输入 up。
        print("可输入 p 变量名 查看变量，输入 n 执行下一行，输入 c 继续。")
        # 进入调试器；p 看变量，n 单步，s 进入函数，c 继续，up 返回调用者栈帧。
        breakpoint()


# 核心逻辑计算函数
# 串起单层模拟计算；默认不打印详细阶段，也不进入断点。这里依赖上面创建的全局模拟张量。
def run_debug(trace=False, pdb_mode=False):
    # 打印阶段提示；此处数据是上面合成的张量，没有实际读取 COCO 图片。
    print(f"数据加载完毕！\n")
    # 显示输入形状，核对 [2,16,4]、[2,80,4,4]、[2,68,4,4]。
    print(f"anchors:{anchors.shape},cls_score:{cls_score.shape},bbox_pred:{bbox_pred.shape}\n")
    # 显示目标形状和分类归一化因子，核对 [2,16]、[2,16,4]、3。
    print(f"labels:{labels.shape},bbox_targets:{bbox_targets.shape},avg_factor:{avg_factor}\n")

    # 调用阶段输出工具；括号内的多行字符串会先拼成消息，再由 trace 开关决定是否打印。
    trace_print(
        # 第一个参数是是否输出详细信息；从命令行 --trace 传来。
        trace,
        # 设置当前阶段标题，便于把运行输出与这一段公式计算对应起来。
        "Step 0 / 输入和监督信息",
        # 展示本例固定的数据规模；这行是说明文字，没有创建或改变张量。
        "N=2, H=W=4, 类别数=80, reg_max=16\n"
        # 说明每位置有80个分类logits和4×17个回归logits。
        "每个位置：80 个分类 logits + 68 个回归 logits\n"
        # 统计正样本并转为文本；:= 同时把统计结果保存到 pos_count。当前合成数据没有 ignore 标签。
        "正样本数量=" + str(int(pos_count := (labels < classes_nums).sum()))
    # 结束上面的多行函数调用；其中各行是同一次调用的参数。
    )
    # 启用--pdb时在阶段间暂停；先输入up回到run_debug栈帧，再用p 变量名查看这里的变量。
    pause_if_requested(pdb_mode, "Step 0：查看原始 cls_score、bbox_pred、labels")

    # 数据展开：对应 GFLHead.loss_by_feat_single 中的 permute 和 reshape。
    # [2,80,4,4] 先重排为 [2,4,4,80]，再展平为 [32,80]；每行是一个位置的所有类别。
    cls_score_flat=cls_score.permute(0,2,3,1).reshape(-1,classes_nums)
    # 同样把回归预测从 [2,68,4,4] 转为 [32,68]，与分类行号一一对应。
    bbox_pred_flat=bbox_pred.permute(0,2,3,1).reshape(-1,(reg_max+1)*4)
    

    #监督数据展开
    # 标签 [2,16] 展平为 [32]；顺序是先第一张图，再第二张图。
    labels_flat=labels.reshape(-1)
    # 目标距离 [2,16,4] 展平为 [32,4]，保留每个位置的四条边。
    bbox_targets_flat=bbox_targets.reshape(-1,4)
    # 参考框占位数据 [2,16,4] 展平为 [32,4]，顺序与预测、标签一致。
    anchors_flat=anchors.reshape(-1,4)

    #提取正样本
    # 建立 32 个布尔值的正样本掩码；当前只有 3 个 True。
    pos_mask=(labels_flat>=0)&(labels_flat<classes_nums)
    # 选正样本的回归 logits，从 [32,68] 得到 [3,68]；背景不参与 GIoU/DFL。
    pos_bbox_pred=bbox_pred_flat[pos_mask]
    # 用相同掩码选目标，得到 [3,4]；第一行对应原 labels[0,0]。
    pos_bbox_targets=bbox_targets_flat[pos_mask]
    # 选对应的参考点占位框，得到 [3,4]。
    pos_anchors=anchors_flat[pos_mask]
    # 取前两列像素 x/y，再除 8，得到特征图单位参考点 [3,2]；当前 x/y 步长相等。
    pos_centers=pos_anchors[:,:2]/stride[0]

    # 调用阶段输出工具；括号内的多行字符串会先拼成消息，再由 trace 开关决定是否打印。
    trace_print(
        # 第一个参数是是否输出详细信息；从命令行 --trace 传来。
        trace,
        # 设置当前阶段标题，便于把运行输出与这一段公式计算对应起来。
        "Step 1 / 展平并提取正样本",
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"cls_score_flat: {tuple(cls_score_flat.shape)}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"bbox_pred_flat: {tuple(bbox_pred_flat.shape)}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"pos_mask 中正样本数量: {int(pos_mask.sum())}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"pos_bbox_pred: {tuple(pos_bbox_pred.shape)}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"pos_centers:\n{pos_centers.detach()}"
    # 结束上面的多行函数调用；其中各行是同一次调用的参数。
    )
    # 启用--pdb时在阶段间暂停；先输入up回到run_debug栈帧，再用p 变量名查看这里的变量。
    pause_if_requested(pdb_mode, "Step 1：查看展平后的预测和正样本")

    # 1) Integral：把每个正样本的 68 个回归 logits 解码成四条边距离。
    # 调用 Integral，把 [3,68] 的回归 logits 转为 [3,4] 的连续距离，可用 F11 进入函数看 softmax 和期望。
    pos_bbox_pred_corners = integral_decode(pos_bbox_pred)
    # 核对结果是 [3,4]，不是 3 个框各自只有一个距离。
    print(f"\n预测四条边距离 shape: {pos_bbox_pred_corners.shape}")

    # 单独重算第一个正样本左边的概率，仅供展示，不用于改变损失。
    first_distribution = F.softmax(
        # 取首行前 17 个 logits，对其做 softmax；detach 让展示张量脱离梯度图。
        pos_bbox_pred[0, :reg_max + 1], dim=0).detach()
    # 调用阶段输出工具；括号内的多行字符串会先拼成消息，再由 trace 开关决定是否打印。
    trace_print(
        # 第一个参数是是否输出详细信息；从命令行 --trace 传来。
        trace,
        # 设置当前阶段标题，便于把运行输出与这一段公式计算对应起来。
        "Step 2 / Integral",
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"第一个正样本左边的17个概率之和: {first_distribution.sum().item():.6f}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"第一个正样本左边概率前5项: {first_distribution[:5]}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"预测四条边距离:\n{pos_bbox_pred_corners.detach()}"
    # 结束上面的多行函数调用；其中各行是同一次调用的参数。
    )
    # 启用--pdb时在阶段间暂停；先输入up回到run_debug栈帧，再用p 变量名查看这里的变量。
    pause_if_requested(pdb_mode, "Step 2：查看 softmax 概率和 Integral 结果")

    # 2) 根据四条边距离得到预测框；真实框也由真实距离目标解码得到。
    # 将预测距离转为 [3,4] 的 xyxy 预测框；仍处于特征图坐标单位。
    pos_decode_bbox_pred = distances_to_boxes(pos_centers, pos_bbox_pred_corners)
    # 将手工真实距离转为同单位 xyxy 真值框；是本脚本的简化输入方式。
    pos_decode_bbox_targets = distances_to_boxes(pos_centers, pos_bbox_targets)

    # 3) 计算预测框和真实框的 IoU。GFL 用它作为 QFL 正样本的质量标签。
    # 生成 QFL 的质量标签，每个正样本对应一个 IoU。
    quality = aligned_iou(
        # 只在标签这条路径 detach 预测框，避免 QFL 通过 IoU 标签优化回归分支；原预测框仍可用于 GIoU 求导。
        pos_decode_bbox_pred.detach(), pos_decode_bbox_targets)
    # 显示 3 个 IoU；它们是当前预测质量，不是永久固定的标注类别。
    print(f"预测框与真实框 IoU: {quality.detach()}")

    # 调用阶段输出工具；括号内的多行字符串会先拼成消息，再由 trace 开关决定是否打印。
    trace_print(
        # 第一个参数是是否输出详细信息；从命令行 --trace 传来。
        trace,
        # 设置当前阶段标题，便于把运行输出与这一段公式计算对应起来。
        "Step 3 / 解码预测框并计算 IoU",
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"预测框 xyxy:\n{pos_decode_bbox_pred.detach()}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"真实框 xyxy:\n{pos_decode_bbox_targets.detach()}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"IoU（作为 QFL 正样本质量标签）: {quality.detach()}"
    # 结束上面的多行函数调用；其中各行是同一次调用的参数。
    )
    # 启用--pdb时在阶段间暂停；先输入up回到run_debug栈帧，再用p 变量名查看这里的变量。
    pause_if_requested(pdb_mode, "Step 3：查看预测框、真实框和 IoU")

    # 分类置信度作为 GIoU/DFL 的样本权重；detach 表示它不反向影响分类分支。
    # 分类 logits 脱离梯度图后 sigmoid，沿 80 类取最大值再选正样本，得到 [3] 回归权重；[0] 是取 max 返回的值，非索引。
    weight_targets = cls_score_flat.detach().sigmoid().max(dim=1)[0][pos_mask]

    # 调用阶段输出工具；括号内的多行字符串会先拼成消息，再由 trace 开关决定是否打印。
    trace_print(
        # 第一个参数是是否输出详细信息；从命令行 --trace 传来。
        trace,
        # 设置当前阶段标题，便于把运行输出与这一段公式计算对应起来。
        "Step 4 / 样本权重",
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"weight_targets（正样本最大分类置信度）: {weight_targets}\n"
        # 强调权重detach后只提供数值，回归损失不沿权重路径影响分类输出。
        "该权重 detach，不会通过回归损失反向更新分类分支。"
    # 结束上面的多行函数调用；其中各行是同一次调用的参数。
    )

    # 4) GIoU Loss：比较最终预测框和真实框，属于框级回归损失。
    # 计算每对预测框和真值框的 GIoU 损失，返回 [3]。
    loss_bbox_each = aligned_giou_loss(
        # 输入没有 detach 的预测框，因此这项损失可沿 Integral 反传到 bbox_pred。
        pos_decode_bbox_pred, pos_decode_bbox_targets)
    # MMDetection 在这里传 avg_factor=1.0，因此是加权和，不再除以权重和。
    # 每个框的 GIoU 损失乘样本置信度后求和；这里只模拟单层局部结果，正式 loss_by_feat 后续还会除全层回归权重和。
    loss_bbox = (loss_bbox_each * weight_targets).sum()

    # 调用阶段输出工具；括号内的多行字符串会先拼成消息，再由 trace 开关决定是否打印。
    trace_print(
        # 第一个参数是是否输出详细信息；从命令行 --trace 传来。
        trace,
        # 设置当前阶段标题，便于把运行输出与这一段公式计算对应起来。
        "Step 5 / GIoU Loss",
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"每个正样本的 GIoU Loss: {loss_bbox_each.detach()}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"加权后的 loss_bbox: {loss_bbox.item():.6f}"
    # 结束上面的多行函数调用；其中各行是同一次调用的参数。
    )
    # 启用--pdb时在阶段间暂停；先输入up回到run_debug栈帧，再用p 变量名查看这里的变量。
    pause_if_requested(pdb_mode, "Step 5：查看 GIoU Loss")

    # 5) DFL：监督四条边的离散概率分布。
    # 把 [3,68] 展为 [12,17]；3 框×4 边，一行是一条边的 logits。
    pred_list=pos_bbox_pred.reshape(-1,(reg_max+1))
    # 把 [3,4] 目标展为 [12]，顺序与 pred_list 每一行对应。
    target_list=pos_bbox_targets.reshape(-1)

    # 取左邻整数；当前目标非负，转 long 等价于向下取整，例如 4.5→4。
    dis_left=target_list.long()
    # 右邻整数比左邻多 1，例如 4→5；它们是候选索引，不是可更新参数。
    dis_right=dis_left+1
    # 左候选权重=右邻整数-真实距离，例如 5-4.5=0.5。
    weight_left=dis_right.float()-target_list
    # 右候选权重=真实距离-左邻整数，例如 4.5-4=0.5；左右和为 1。
    weight_right=target_list-dis_left.float()

    # 每条边计算 -log p_left，保留 [12]；函数内部做 log_softmax，输入必须是 logits。
    loss_left=F.cross_entropy(pred_list,dis_left,reduction='none')
    # 同样计算 -log p_right，保留 [12]；与左项使用同一组 logits。
    loss_right=F.cross_entropy(pred_list,dis_right,reduction='none')
    # 一个正样本有四条边，因此每条边复制该样本的 weight_targets。
    # [3]→[3,1]→[3,4]→[12]，每个正样本的权重复制给四条边；这不同于 DFL 左右插值权重。
    edge_weights = weight_targets[:, None].expand(-1, 4).reshape(-1)
    # 得到每条边的 DFL=-w_left log p_left-w_right log p_right，形状 [12]。
    dfl_each = loss_left * weight_left + loss_right * weight_right
    # MMDetection 的 avg_factor=4.0，对四条边的损失做平均。
    # 乘框级权重后求和，再除 4 平均四边；这是单层局部值，正式多层汇总还有权重和归一化。
    loss_dfl=(dfl_each * edge_weights).sum() / 4.0

    # 调用阶段输出工具；括号内的多行字符串会先拼成消息，再由 trace 开关决定是否打印。
    trace_print(
        # 第一个参数是是否输出详细信息；从命令行 --trace 传来。
        trace,
        # 设置当前阶段标题，便于把运行输出与这一段公式计算对应起来。
        "Step 6 / DFL",
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"第一个边目标距离: {target_list[0].item():.4f}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"左右整数: {dis_left[0].item()}, {dis_right[0].item()}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"左右权重: {weight_left[0].item():.4f}, {weight_right[0].item():.4f}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"对应左右交叉熵: {loss_left[0].item():.6f}, {loss_right[0].item():.6f}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"DFL: {loss_dfl.item():.6f}"
    # 结束上面的多行函数调用；其中各行是同一次调用的参数。
    )
    # 启用--pdb时在阶段间暂停；先输入up回到run_debug栈帧，再用p 变量名查看这里的变量。
    pause_if_requested(pdb_mode, "Step 6：查看 DFL 的左右插值")

    # 6) QFL：分类分支的正样本目标是 IoU，负样本目标是0。
    # 建立 [32] 浮点质量标签，全零对应背景；不要用整数类型保存 IoU。
    quality_all = torch.zeros_like(labels_flat, dtype=cls_score_flat.dtype)
    # 把 3 个正样本位置的质量填为 IoU，其他 29 个背景位置保持 0。
    quality_all[pos_mask] = quality
    # 输入 [32,80] logits、[32] 类别和质量，得到标量 QFL；可 F11 进入看真实类别目标覆盖。
    loss_cls = quality_focal_loss(cls_score_flat, labels_flat, quality_all)

    # 为展示重新提取正样本行号 [0,5,18]；不改变训练结果。
    positive_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
    # 取展示用正样本类别 [1,1,0]。
    positive_labels = labels_flat[positive_indices]
    # 为展示计算分类概率，并准备按行号和真实类别成对索引。
    positive_probabilities = cls_score_flat.sigmoid()[
        # 成对索引取每个正样本真实类别的概率；不是取所有行列组合。
        positive_indices, positive_labels]
    # 调用阶段输出工具；括号内的多行字符串会先拼成消息，再由 trace 开关决定是否打印。
    trace_print(
        # 第一个参数是是否输出详细信息；从命令行 --trace 传来。
        trace,
        # 设置当前阶段标题，便于把运行输出与这一段公式计算对应起来。
        "Step 7 / QFL",
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"正样本类别: {positive_labels}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"正样本类别预测概率: {positive_probabilities.detach()}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"正样本 IoU 质量标签: {quality.detach()}\n"
        # 格式化展示当前阶段的形状或中间数值；detach只断开展示副本的梯度，item把标量转为普通数，均不修改原计算结果。
        f"QFL: {loss_cls.item():.6f}"
    # 结束上面的多行函数调用；其中各行是同一次调用的参数。
    )
    # 启用--pdb时在阶段间暂停；先输入up回到run_debug栈帧，再用p 变量名查看这里的变量。
    pause_if_requested(pdb_mode, "Step 7：查看 QFL 的类别概率和 IoU 目标")

    # 配置中的三个 loss_weight：QFL=1，GIoU=2，DFL=0.25。
    # 按配置系数汇总局部损失，保留完整梯度图；这里回归项尚未做正式多层汇总归一化，不能直接当训练日志的相同数值。
    total_loss = loss_cls + 2.0 * loss_bbox + 0.25 * loss_dfl

    # item() 将单元素张量转成 Python 数，仅用于显示；原 loss_cls 仍保留梯度图。
    print(f"loss_cls (QFL): {loss_cls.item():.6f}")
    # 显示未乘 2.0 的单层 GIoU 加权和，保留小数点后六位。
    print(f"loss_bbox (GIoU): {loss_bbox.item():.6f}")
    # 显示未乘 0.25 的单层 DFL 加权值。
    print(f"loss_dfl (DFL): {loss_dfl.item():.6f}")
    # 显示已乘三项系数的总损失；之后 backward 必须使用 total_loss 张量本身，不能对 item() 求导。
    print(f"total_loss: {total_loss.item():.6f}")

    # 用于展示完整训练链路：三项损失汇总后反向传播，更新 cls_score/bbox_pred。
    # 建立 SGD，学习率 0.01；此教学脚本直接优化两个叶子 logits。真实训练通常优化 model.parameters() 的网络权重。
    optimizer = torch.optim.SGD([cls_score, bbox_pred], lr=0.01)
    # 把更新前第一个分类 logit 保存为普通数值作对比；detach/flatten/item 不参与损失求导。
    old_cls_logit = cls_score.detach().flatten()[0].item()
    # 保存更新前第一个回归 logit，方便观察 step 后是否变化。
    old_bbox_logit = bbox_pred.detach().flatten()[0].item()
    # 清空优化器所管理张量的旧梯度；PyTorch 默认会累加梯度，普通单步训练需要清空。
    optimizer.zero_grad()
    # 沿计算图用链式法则求导，将梯度累加到 cls_score.grad 和 bbox_pred.grad；此行不更新它们的数值。
    total_loss.backward()
    # 显示分类梯度绝对值均值；abs 防止正负抵消，.6e 以科学计数法显示，非损失大小。
    print(f"cls_score 梯度均值: {cls_score.grad.abs().mean().item():.6e}")
    # 显示回归梯度绝对值均值；本脚本回归梯度来自 GIoU 和 DFL，QFL 的 IoU 标签已 detach。
    print(f"bbox_pred 梯度均值: {bbox_pred.grad.abs().mean().item():.6e}")
    # 执行 SGD：新值=旧值-0.01×梯度；修改被管理的 logits，不会自动重新计算旧 loss 或中间张量。
    optimizer.step()
    # 比较第一个分类 logit 更新前后；展示的 [0] 是展平后的第一个元素。
    print(f"cls_score[0] 更新: {old_cls_logit:.6f} -> {cls_score.detach().flatten()[0].item():.6f}")
    # 比较第一个回归 logit 更新前后；要看到新预测框和新损失，必须用新 logits 重新计算。
    print(f"bbox_pred[0] 更新: {old_bbox_logit:.6f} -> {bbox_pred.detach().flatten()[0].item():.6f}")
    # 启用--pdb时在阶段间暂停；先输入up回到run_debug栈帧，再用p 变量名查看这里的变量。
    pause_if_requested(pdb_mode, "Step 8：查看梯度和 optimizer.step() 后的参数")
    # 返回本轮更新前算出的损失张量；step 后旧 loss 不会刷新，且本轮 backward 已使用其计算图。
    return total_loss


# 仅直接运行此文件时执行入口；被 import 时不执行此入口，但上面的全局数据创建仍会执行。
if __name__ == '__main__':
    # 创建命令行解析器，后面的 description 是 --help 时的说明。
    parser = argparse.ArgumentParser(
        # 设置脚本用途说明；这是字符串配置，没有参与数学计算。
        description='逐步展示 GFL 的 Integral、QFL、GIoU 和 DFL 计算链路')
    # 注册一个可选命令行参数。
    parser.add_argument(
        # 出现 --trace 时置 True；未提供时为 False，不需要额外写 True 字符串。
        '--trace', action='store_true',
        # 提供 --trace 的帮助文本。
        help='打印每个阶段的关键张量、形状和中间结果')
    # 注册一个可选命令行参数。
    parser.add_argument(
        # 出现 --pdb 时将 args.pdb_mode 置 True；dest 指定属性名。
        '--pdb', dest='pdb_mode', action='store_true',
        # 提供 --pdb 帮助文本；暂停在辅助函数时，先 up 才能看调用者局部变量。
        help='在每个阶段进入 Python 调试器；输入 n 单步，c 继续，p 变量查看')
    # 读取终端参数并保存到 args；不带参数时两个开关都是 False。
    args = parser.parse_args()
    # 把命令行选项传入主流程，执行一次模拟前向、损失计算、反向和 logits 更新。
    run_debug(trace=args.trace, pdb_mode=args.pdb_mode)






