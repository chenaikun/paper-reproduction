import torch





def debug():
    T_p=torch.tensor([[0.05,0.10,0.60,0.20,0.05]],dtype=torch.float32)
    S_l=torch.zeros((1,5),requires_grad=True)
    S_p=torch.softmax(S_l,dim=-1)
    S_l_log=torch.log_softmax(S_l,dim=-1)
    kl=torch.nn.functional.kl_div(S_l_log,T_p,reduction="batchmean")
    optimzer=torch.optim.SGD(S_l,lr=0.1)
    kl.backward()
    optimzer.step()
    print(f"T_p: {T_p},S_p: {S_p}")
    print(f"S_p_sum: {S_p.sum(dim=-1)}")
    print(f"KL: {kl}")
    print(f"s_l.grad: {S_l.grad}")





if __name__ == '__main__':
    debug()