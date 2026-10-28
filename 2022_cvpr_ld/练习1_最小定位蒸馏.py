import torch





def debug():
    T_p=torch.tensor([[[0.05,0.10,0.60,0.20,0.05],
                      [0.10, 0.60, 0.20, 0.05, 0.05],
                      [0.05, 0.20, 0.60, 0.10, 0.05],
                      [0.05, 0.10, 0.20, 0.60, 0.05]]],dtype=torch.float32)
    S_l=torch.zeros((3,4,5),requires_grad=True)
    S_p=torch.softmax(S_l,dim=-1)
    optimzer=torch.optim.SGD([S_l],lr=0.1)
    #构造位置掩码
    pos_mask=torch.tensor([True,False,True])
    i=0
    #S_l_log=torch.log_softmax(S_l,dim=-1)
    #KL=torch.nn.functional.kl_div(S_l_log,T_p,reduction="none").sum(dim=-1).mean()
    #print(f"KL: {KL}")

    # print(f"T=1: {torch.softmax(T_l/1,dim=-1)},{torch.log_softmax(S_l/1,dim=-1)},{torch.nn.functional.kl_div(torch.log_softmax(S_l/1,dim=-1),torch.softmax(T_l/1,dim=-1),reduction='batchmean')}")
    # print(f"T=2: {torch.softmax(T_l/2,dim=-1)},{torch.log_softmax(S_l/2,dim=-1)},{torch.nn.functional.kl_div(torch.log_softmax(S_l/2,dim=-1),torch.softmax(T_l/2,dim=-1),reduction='batchmean')}")
    # print(f"T=3: {torch.softmax(T_l/3,dim=-1)},{torch.log_softmax(S_l/2,dim=-1)},{torch.nn.functional.kl_div(torch.log_softmax(S_l/3,dim=-1),torch.softmax(T_l/3,dim=-1),reduction='batchmean')}")
    while i<1:


        S_p=torch.softmax(S_l,dim=-1)


        optimzer.zero_grad()
        S_l_log=torch.log_softmax(S_l,dim=-1)
        KL=torch.nn.functional.kl_div(S_l_log,T_p,reduction="none").sum(dim=-1).mean(dim=-1)
        KL=KL[pos_mask].mean()
        KL.backward()
        optimzer.step()

        # S_l_new=S_l
        # S_p_new=torch.softmax(S_l_new,dim=-1)
        # S_l_log_new=torch.log_softmax(S_l_new,dim=-1)
        # kl_new=torch.nn.functional.kl_div(S_l_log_new,T_p,reduction="batchmean")

        print(f"T_p: {T_p},S_p: {S_p}")
        print(f"S_p_sum: {S_p.sum(dim=-1)}")
        print(f"KL: {KL}")
        print(f"s_l.grad: {S_l.grad}")
        # if i%1000==0 :
        #     print(f"T_p: {T_p},S_p: {S_p}")
        #     print(f"S_p_sum: {S_p.sum(dim=-1)}")
        #     print(f"KL: {KL}")
        #     print(f"s_l.grad: {S_l.grad}")

        # print(f"S_l_new: {S_l_new}")
        # print(f"T_p: {T_p},S_p_new: {S_p_new}")
        # print(f"S_p_sum_new: {S_p_new.sum(dim=-1)}")
        # print(f"KL_new: {kl_new}")
        i=i+1
    error = torch.max(torch.abs(S_p - T_p))
    print(f"error: {error}")





if __name__ == '__main__':
    debug()
