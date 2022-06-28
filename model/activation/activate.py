
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationFunc(nn.Module):
    def __init__(self,name_func=None):
        super(ActivationFunc,self).__init__()
        if name_func == None or name_func=="":
            self.func = nn.Identity()
        elif name_func == "softmax":
            self.func == nn.Softmax()
        elif name_func == "sigmoid":
            self.func = nn.Sigmoid()
        elif name_func=="ReLu":
            self.func = nn.ReLU()
        elif name_func == "Gelu":
            self.func = nn.GELU()
        elif name_func == "Mish":
            self.func = nn.Mish()
        elif name_func == "lekyReLU":
            self.func = nn.LeakyReLU(negative_slope=0.1)
    def forward(self, x):
        return self.func(x)