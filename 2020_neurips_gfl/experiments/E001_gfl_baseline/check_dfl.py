import torch
import torch.nn.functional as F

#pred = torch.zeros(1,5,requires_grad=True)
model = torch.nn.Linear(1,5,bias=False)
torch.nn.init.zeros_(model.weight)
x=torch.tensor([[1.0]])
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

#真实距离
label = torch.tensor([2.8])

dis_left = label.long()
dis_right=dis_left+1
weight_left=dis_right.float()-label
weight_right=label-dis_left.float()


for step in range(1000):
    pred=model(x)
#pred=model(x)
    pred.retain_grad()
    


    probs=F.softmax(pred,dim=1)
    print(f"预测概率：{probs}")
    print(f"左右权重: {weight_left},{weight_right}")


    ce_left=F.cross_entropy(pred,dis_left,reduction="none")
    loss_left=ce_left*weight_left

    ce_right=F.cross_entropy(pred,dis_right,reduction="none")
    loss_right=ce_right*weight_right

    #总损失
    loss_dfl = loss_left+loss_right
    print(f"左交叉熵：{ce_left}")
    print(f"左加权损失：{loss_left}")

    print(f"右交叉熵：{ce_right}")
    print(f"右加权损失：{loss_right}")


    print(f"DFL: {loss_left+loss_right}")



    #执行反向传播
    optimizer.zero_grad()
    loss_dfl.mean().backward()
    print(f"logits的梯度：{pred.grad}")
    print(f"网络权重的梯度：{model.weight.grad}")




    #手动更新，测试损失是否下降
    # lr=0.1

    # with torch.no_grad():
    #     pred-=lr*pred.grad
    optimizer.step()
    with torch.no_grad():
        new_pred=model(x)
        new_probs=F.softmax(new_pred,dim=1)

    print(f"更新后的权重:{model.weight.detach()}")
    print(f"重新计算的logits:{new_pred}")
    print(f"重新计算的概率：{new_probs}")