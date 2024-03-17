
import numpy as np

class myBN:
    def __init__(self,momentum=0.01,eps=1e-5,feat_dim=2) -> None:
        
        self.running_mean=np.zeros(shape=(feat_dim,))
        self.running_var=np.ones(shape=(feat_dim,))

        self.momentum=momentum
        self.eps=eps

        self.beta=np.zeros(shape=(feat_dim,))
        self.gamma=np.ones(shape=(feat_dim,))

    def batch_norm(self,x):
        if self.training :
            x_mean=x.mean(axis=0)
            x_var=x.var(axis=0)
            self.running_mean=(1-self.momentum)*x_mean+self.momentum*self.running_mean
            self.running_var=(1-self.momentum)*x_var+self.momentum*self.running_var

            x_hat=(x-x_mean)/np.sqrt(x_var+self.eps)
        else:
            x_hat=(x-self.running_mean)/np.sqrt(self.running_var+self.eps)


        return self.gamma*x_hat+self.beta