import torch
import torch.nn.functional as F

pred = torch.zeros(1,5)

#真实距离
label = torch.tensor([2.8])

dis_left = label.long()
dis_right=dis_left+1

weight_left=dis_right.float()-label
weigth_right=label-dis_left.float()


probs=F.softmax(pred,dim=1)
print(f"预测概率：{probs}")
print(f"左右权重: {weight_left},{weigth_right}")


ce_left=F.cross_entropy(pred,dis_left,reduction="none")
loss_left=ce_left*weight_left

ce_right=F.cross_entropy(pred,dis_right,reduction="none")
loss_right=ce_right*weigth_right
print(f"左交叉熵：{ce_left}")
print(f"左加权损失：{loss_left}")

print(f"右交叉熵：{ce_right}")
print(f"右加权损失：{loss_right}")


print(f"DFL: {loss_left+loss_right}")