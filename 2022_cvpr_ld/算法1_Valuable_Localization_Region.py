import torch


# 论文Algorithm 1 Valuable Localization Region的代码实现

#计算IoU

#框A:（0,0,4,4） 框B：（2,0,6,4）

A=torch.tensor([0,0,4,4])
B=torch.tensor([0,0,4,4])

#交集框：[2,0,4,4]

#判断是否相交，并返回相交面积，以及相交的框的左上和右下坐标
def is_intersect(A: torch.Tensor,B: torch.Tensor):
        A_B=torch.zeros(4)
        
    #判断有没有交集
        A_x=[A[0],A[2]]
        A_y=[A[1],A[3]]
    
        B_x=[B[0],B[2]]
        B_y=[B[1],B[3]]
        A_B[0]=max(A_x[0],B_x[0])
        A_B[1]=max(A_y[0],B_y[0])
        A_B[2]=min(A_x[1],B_x[1])
        A_B[3]=min(A_y[1],B_y[1])  
        w=A_B[2]-A_B[0]
        h=A_B[3]-A_B[1]

        if w<=0 or h<=0:
              return False,0.0,A_B
        else:
              return True,w*h,A_B


def calculate_iou_giou_diou(A: torch.Tensor,B: torch.Tensor):
    #S_a,S_b为面积
    S_a=abs((A[2]-A[0])*(A[3]-A[1]))
    S_b=abs((B[2]-B[0])*(B[3]-B[1]))
    A_c=torch.tensor([A[0]+(A[2]-A[0])/2.0,A[1]+(A[3]-A[1])/2.0])
    B_c=torch.tensor([B[0]+(B[2]-B[0])/2.0,B[1]+(B[3]-B[1])/2.0])
    is_inter,S_AB,AB=is_intersect(A,B)
    IoU=S_AB/(S_a+S_b-S_AB)
    #计算最小外接矩形坐标
    min_AB=torch.tensor([min(A[0],B[0]),min(A[1],B[1]),max(A[2],B[2]),max(A[3],B[3])])
    #计算GIoU
    S_a_or_b=S_a+S_b-S_AB
    GIoU=IoU-(abs((min_AB[2]-min_AB[0])*(min_AB[3]-min_AB[1]))-S_a_or_b)/abs((min_AB[2]-min_AB[0])*(min_AB[3]-min_AB[1]))
    p=(B_c[1]-A_c[1])**2+(B_c[0]-A_c[0])**2
    c=(min_AB[2]-min_AB[0])**2+(min_AB[3]-min_AB[1])**2
    DIoU=IoU-p/c

    return [IoU,GIoU,DIoU]

def calculate_DIoU_matrix(anchors: list,gts: list):
     i=len(anchors)
     j=len(gts)
     DIoU_matrix=torch.zeros((i,j))
     for m in range(i):
        for n in range(j):
            DIoU_matrix[m][n]=calculate_iou_giou_diou(anchors[m],gts[n])[2]

     return DIoU_matrix

if __name__ == '__main__':
    anchors=[torch.tensor([2,3,4,6]),torch.tensor([1,5,2,8]),torch.tensor([1,4,7,5])]
    gts=[torch.tensor([2,5,6,8]),torch.tensor([1,7,5,9])]
   
    DIoU_matrix=calculate_DIoU_matrix(anchors,gts)
    print(f"DIoU_matrix:{DIoU_matrix}")