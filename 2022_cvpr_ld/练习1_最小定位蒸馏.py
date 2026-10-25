import torch





def debug():
    T_p=torch.tensor([[0.05,0.10,0.60,0.20,0.05]],dtype=torch.float32)
    S_l=torch.zeros((1,5),requires_grad=True)
    S_p=torch.softmax(S_l,dim=-1)
    optimzer=torch.optim.SGD([S_l],lr=0.1)
    i=0
    while i<10000:

        optimzer.zero_grad()
        S_p=torch.softmax(S_l,dim=-1)
        S_l_log=torch.log_softmax(S_l,dim=-1)
        kl=torch.nn.functional.kl_div(S_l_log,T_p,reduction="batchmean")
        
        kl.backward()
        optimzer.step()

        # S_l_new=S_l
        # S_p_new=torch.softmax(S_l_new,dim=-1)
        # S_l_log_new=torch.log_softmax(S_l_new,dim=-1)
        # kl_new=torch.nn.functional.kl_div(S_l_log_new,T_p,reduction="batchmean")


        if i%1000==0 :
            print(f"T_p: {T_p},S_p: {S_p}")
            print(f"S_p_sum: {S_p.sum(dim=-1)}")
            print(f"KL: {kl}")
            print(f"s_l.grad: {S_l.grad}")

        # print(f"S_l_new: {S_l_new}")
        # print(f"T_p: {T_p},S_p_new: {S_p_new}")
        # print(f"S_p_sum_new: {S_p_new.sum(dim=-1)}")
        # print(f"KL_new: {kl_new}")
        i=i+1
    error = torch.max(torch.abs(S_p - T_p))
    print(f"error: {error}")





if __name__ == '__main__':
    debug()