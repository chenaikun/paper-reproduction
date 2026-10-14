import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# ----------------- 1. 解决目录路径导入问题 -----------------
# 获取当前脚本所在目录 (test_debug) 的上一级目录 (2020_neurips_gfl)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 将根目录和 mmdetection 路径加入到 sys.path 中
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

mmdet_path = PROJECT_ROOT / "mmdetection"
if mmdet_path.exists() and str(mmdet_path) not in sys.path:
    sys.path.insert(0, str(mmdet_path))

# 验证导入 (根据你的实际类名导入，如果是官方 gfl 可直接导入)
try:
    from mmdet.models.dense_heads import GFLHead
    print("成功导入 MMDetection 模块！")
except ImportError as e:
    print(f"导入提示: {e}，将直接进行纯张量逻辑调试。")


#基础数据

N=2 #batchsize为2
classes_nums=80
H,W=4,4 
reg_max=16  #最大离散值
anchors_num=H*W  #一张特征图全部特征点
stride=(8,8) #下采样倍数


#构造head预测值
cls_score=torch.randn(N,classes_nums,H,W,requires_grad=True)

bbox_pred=torch.randn(N,4*(reg_max+1),H,W,requires_grad=True)

#构造几何基准
y_grid,x_grid=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
x_centers=(x_grid.float()+0.5)*stride[0]
y_centers=(y_grid.float()+0.5)*stride[1]
points=torch.stack([x_centers.flatten(),y_centers.flatten()],dim=-1)
#.unsqueeze(0):把这叠卡片装进 1 个文件夹里 ---> 得到 [1 个文件夹, 16 张卡片, 4 个坐标]
#.repeat(2, 1, 1):按 (2, 1, 1) 放到复印机里 ---> 得到 [2 个文件夹, 16 张卡片, 4 个坐标](正好对应 Batch 中的 2 张图片，每张图片各分到一个装有 16 张卡片的文件夹)
anchors=torch.cat([points,points],dim=-1).unsqueeze(0).repeat(N,1,1)

#构造监督信息
labels=torch.full((N,anchors_num),classes_nums,dtype=torch.long)  #类别数设置为80  一般用于当作负样本或者背景


#加入几个正样本点
labels[0,5]=1   #第一张图的第6个点类别为1  
labels[1,2]=0   #第二张图的第3个点类别为0
labels[0,0]=1   #第一张图的第1个点类别为1

label_weights = torch.ones((N,anchors_num),dtype=torch.float32)

#正样本回归目标
bbox_targets=torch.zeros((N,anchors_num,4),dtype=torch.float32)
bbox_targets[0,5]=torch.tensor([3.2, 5.1, 4.0, 2.5])
bbox_targets[1,2]=torch.tensor([8.6, 2.3, 7.4, 6.1])
bbox_targets[0,0]=torch.tensor([4.5, 4.5, 3.0, 3.0])

#控制正样本数量的兜底
#防止正样本出现数据为0
avg_factor=max(int((labels<classes_nums).sum()),1)


#核心逻辑计算函数
def run_debug():
    print(f"数据加载完毕！\n")
    print(f"anchors:{anchors.shape},cls_score:{cls_score.shape},bbox_pred:{bbox_pred.shape}\n")
    print(f"labels:{labels.shape},bbox_targets:{bbox_targets.shape},avg_factor:{avg_factor}\n")

    #数据展开
    cls_score_flat=cls_score.permute(0,2,3,1).reshape(-1,classes_nums)
    bbox_pred_flat=bbox_pred.permute(0,2,3,1).reshape(-1,(reg_max+1)*4)
    #监督数据展开
    labels_flat=labels.reshape(-1)
    bbox_targets_flat=bbox_targets.reshape(-1,4)

    #提取正样本
    pos_mask=(labels_flat>=0)&(labels_flat<classes_nums)
    pos_bbox_pred=bbox_pred_flat[pos_mask]
    pos_bbox_targets=bbox_targets_flat[pos_mask]


    #模拟计算DFL loss
    pred_list=pos_bbox_pred.reshape(-1,(reg_max+1))
    target_list=pos_bbox_targets.reshape(-1)

    dis_left=target_list.long()
    dis_right=dis_left+1
    weight_left=dis_right.float()-target_list
    weight_right=target_list-dis_left.float()

    loss_left=F.cross_entropy(pred_list,dis_left,reduction='none')
    loss_right=F.cross_entropy(pred_list,dis_right,reduction='none')
    loss_dfl=(loss_left*weight_left+loss_right*weight_right).sum()/avg_factor

    print(f"计算成功 loss_dfl: {loss_dfl}")
    return loss_dfl


if __name__ == '__main__':
    run_debug()



