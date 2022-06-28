from turtle import forward
from typing import List
import torch
import torch.nn as nn
import torch.nn. functional as F
from model.activation.activate import ActivationFunc
from model.gcn_block.attention_block import Channel_Attention_Block,Position_Attention_Block
#from model.decoder.deeplabv3 import DeepLabV3Decoder

class ConvBlock(nn.Sequential):
    def __init__(self,in_channels,out_channels,kernel_size,strides=1,padding=0,activation=None):
        #super(ConvBlock,self).__init__()
        conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                            stride=strides,padding=padding)
        bn   = nn.BatchNorm2d(out_channels)
        act  = ActivationFunc(name_func=activation)
        super(ConvBlock,self).__init__(conv,bn,act)


"""
class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int) -> None:
        super(FPN,self).__init__()
      

        self.output1 = ConvBlock(in_channels=in_channels_list[0],out_channels=out_channels,kernel_size=1,
                            strides=1,padding=0,activation="lekyReLU")
        self.output2 =ConvBlock(in_channels=in_channels_list[1],out_channels=out_channels,kernel_size=1,
                            strides=1,padding=0,activation="lekyReLU")
        self.output3 = ConvBlock(in_channels=in_channels_list[2],out_channels=out_channels,kernel_size=1,
                            strides=1,padding=0,activation="lekyReLU")

        self.merge1 = ConvBlock(in_channels=out_channels,out_channels=out_channels,kernel_size=1,
                            strides=1,padding=0,activation="lekyReLU")
        self.merge2 = ConvBlock(in_channels=out_channels,out_channels=out_channels,kernel_size=1,
                            strides=1,padding=0,activation="lekyReLU")

    def forward(self, x :List) -> List[torch.Tensor]:
        

        output1 = self.output1(x[0])
        output2 = self.output2(x[1])
        output3 = self.output3(x[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return [output1, output2, output3]

"""

class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int) -> None:
        super(FPN,self).__init__()
      

        self.output1 = Channel_Attention_Block(n_feat=in_channels_list[0],out_channels=out_channels)
        self.output2 = Channel_Attention_Block(n_feat=in_channels_list[1],out_channels=out_channels)
        self.output3 = Channel_Attention_Block(n_feat=in_channels_list[2],out_channels=out_channels)

        self.merge1 = ConvBlock(in_channels=out_channels,out_channels=out_channels,kernel_size=1,
                            strides=1,padding=0,activation="lekyReLU")
        self.merge2 = ConvBlock(in_channels=out_channels,out_channels=out_channels,kernel_size=1,
                            strides=1,padding=0,activation="lekyReLU")

    def forward(self, x) -> List[torch.Tensor]:
        
        output1 = self.output1(x[0])
        output2 = self.output2(x[1])
        output3 = self.output3(x[2])
       
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return [output1, output2, output3]
