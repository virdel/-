import torch
import torch.nn as nn
import numpy as np

## 默认行向量
class scaleDotProductAttention(nn.Module):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale=scale
        self.softmax=nn.Softmax(dim=2)
    
    def forward(self,q,k,v,mask=None):
        ## q:batch*n_q*d_q
        ## k:batch*n_k*d_k
        ## v:batch*n_v*d_v
        ## dq和dk应该相等
        ## nk和nv
        # u=torch.bmm(q,k.transpose(1,2))
        u=torch.matmul(q,k.transpose(1,2))
        u=u/self.scale
        if mask is not None:
            u=u.masked_fill(mask,-np.inf)

        attn=self.softmax(u) ## batch*nq*nk
        output=torch.bmm(attn,v) ## batch*nq*dv

        return attn,output






if __name__=="__main__":
    m=nn.Softmax(dim=0)
    input=torch.randn(2,3)
    output=m(input)
    print(output)

    batch=10

    n_q, n_k, n_v = 2, 4, 4
    d_q, d_k, d_v = 128, 128, 64

    q=torch.randn(batch,n_q,d_q)
    k=torch.randn(batch,n_k,d_k)
    v=torch.randn(batch,n_v,d_v)

    mask=torch.zeros(batch,n_q,n_k).bool()

    m=scaleDotProductAttention(scale=np.power(d_k,0.5))

    attn,out=m(q,k,v,mask)

    print(attn.shape)

    print(out.shape)


