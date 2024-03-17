import torch
import torch.nn.functional  as F
import numpy as np
## 二维卷积
def conv2d(img,in_channels,out_channels,kernels,bias,stride=1,padding=0):
    N,C,H,W=img.shape
    kh,kw=kernels.shape
    p=padding

    if p:
        img=np.pad(img,((0,0),(0,0),(p,p),(p,p)))
    out_h=(H+2*padding-kh)//stride+1
    out_W=(W+2*padding-kw)//stride+1

    outputs=np.zeros([N,out_channels,out_h,out_W])
    for n in range(N):
        for out in range(out_channels):
            for i in range(in_channels):
                for h in range(out_h):
                    for w in range(out_W):
                        for x in range(kh):
                            for y in range(kw):
                                outputs[n][out][h][w]+=img[n][i][h*stride+x][h*stride+y]*kernels[x][y]

                if i==in_channels-1:
                    outputs[n][out][:][:]+=bias[n][out]
    return outputs




class MyCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1=torch.nn.Conv2d(1,10,5) ## 
        self.conv2=torch.nn.Conv2d(10,20,5)

        self.pooling=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(16820,10)

    def forward(self,x):
        
        batch_size=x.size(0)
        print(x.size())
        x=F.relu(self.pooling(self.conv1(x)))
        print(x.size())
        x=F.relu(self.pooling(self.conv2(x)))
        print(x.size())

        x=x.view(batch_size,-1)

        x=self.fc(x)
        return x

if __name__ == "__main__":
    batch=10

    x = torch.randn(batch, 1,128, 128)

    mycnn=MyCNN()

    y=mycnn(x)
    print(y)





