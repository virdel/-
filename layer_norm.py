import torch
from torch import nn

class LN(nn.Module):
    def __init__(self,normalized_shape,eps=1e-5,elementwise_affine=True) -> None:
        super().__init__()
        self.normalized_shape=normalized_shape
        self.eps=eps
        self.elementwise_affine=elementwise_affine
        if elementwise_affine:
            self.gain=nn.Parameter(torch.ones(normalized_shape))
            self.bias=nn.Parameter(torch.zeros(normalized_shape))

    def forward(self,x):## b*c*w*h
        dims=[-(i+1) for i in range(len(self.normalized_shape))]

        mean=x.mean(dim=dims,keepdims=True) ## b*1*1
        mean_x2=(x**2).mean(dim=dims,keepdims=True)

        var=mean_x2-mean**2

        print("var shape",var.size())
        x_norm=(x-mean)/torch.sqrt(var+self.eps)
        if self.elementwise_affine:
            print("self.gain shape",self.gain.size())
            print("x_norm shape",x_norm.size())
            ## 按位置相乘，存在广播机制
            x_norm=self.gain*x_norm+self.bias
        return x_norm



 
if __name__ == '__main__':
 
    x = torch.linspace(0, 23, 24, dtype=torch.float32)  # 构造输入层
    x = x.reshape([2,3,2*2])  # [b,c,w*h]
    print(x)
    print(x.mean(axis=0).size())
    # 实例化
    print(x.shape[1:])
    ln = LN(x.shape[1:])
    # 前向传播
    x = ln(x)
    print(x.shape)

