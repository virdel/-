import torch
import torch.nn as nn
import numpy as np



## 应该是默认行向量
class scaleDotProductAttention(nn.Module):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale=scale
        self.softmax=nn.Softmax(dim=2)
    
    def forward(self,q,k,v,mask=None):
        ## q:b*n*m  k:b*n*m

        u=torch.bmm(q,k.transpose(1,2))
        u=u/self.scale
        if mask is not None:
            u=u.masked_fill(mask,-np.inf)

        attn=self.softmax(u)
        output=torch.bmm(attn,v)

        return attn,output


## 多头自注意机制
class MultiHeadAttention(nn.Module):

    def __init__(self,n_head,d_k_,d_v_,d_k,d_v,d_o):
        super().__init__()
        self.n_head=n_head
        self.d_k=d_k
        self.d_v=d_v


        self.fc_q=nn.Linear(d_k_,n_head*d_k)
        self.fc_k=nn.Linear(d_k_,n_head*d_k)
        self.fc_v=nn.Linear(d_v_,n_head*d_v)

        self.attention=scaleDotProductAttention(scale=np.power(d_k,0.5))

        self.fc_o=nn.Linear(n_head*d_v,d_o)

    def forward(self,q,k,v,mask=None):

        n_head,d_q,d_k,d_v=self.n_head, self.d_k, self.d_k, self.d_v

        batch,n_q,d_q=q.size()
        batch,n_k,d_k=k.size()
        batch,n_v,d_v=v.size()

        q=self.fc_q(q)  ## batch*d_k*n_head
        k=self.fc_k(k)
        v=self.fc_v(v)

        q=q.view(batch,n_q,n_head,d_q).permute(2,0,1,3).contiguous().view(-1,n_q,d_q)

        k=k.view(batch,n_k,n_head,d_k).permute(2,0,1,3).contiguous().view(-1,n_k,d_k)

        v=v.view(batch,n_v,n_head,d_v).permute(2,0,1,3).contiguous().view(-1,n_v,d_v)

        if mask is not None:
            mask=mask.repeat(n_head,1,1)

        attn,output=self.attention(q,k,v,mask)

        output=output.view(n_head,batch,n_q,d_v).permute(1,2,0,3).contiguous().view(batch,n_q,-1)

        output=self.fc_o(output)

        return attn,output




if __name__=="__main__":
    m=nn.Softmax(dim=0)
    input=torch.randn(2,3)
    output=m(input)
    print(output)
    batch=128
    n_q, n_k, n_v = 2, 4, 4
    d_q_, d_k_, d_v_ = 128, 128, 64

    q = torch.randn(batch, n_q, d_q_)
    k = torch.randn(batch, n_k, d_k_)
    v = torch.randn(batch, n_v, d_v_)    
    mask = torch.zeros(batch, n_q, n_k).bool()

    mha=MultiHeadAttention(n_head=8,d_k_=128,d_v_=64,d_k=128,d_v=64,d_o=128)

    attn,output=mha(q,k,v,mask)

    
    print(attn.size())
    print(output.size())

