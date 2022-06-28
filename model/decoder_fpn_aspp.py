
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.model import SegmentationModel
from model.head.segment_head import SegmentationHead
from model.neck.fpn import FPN
from model.decoder.deeplabv3 import DeepLabV3Decoder
from model.gcn_block.attention_block import Channel_Attention_Block,Position_Attention_Block

"""
x = torch.rand(1,256,56,56)
y = torch.rand(1,512,28,28)
z = torch.rand(1,1024,14,14)
list_feat =[z,y,x]

net = FPN(in_channels_list=[1024,512,256],out_channels=256)

b = net(list_feat)

x = b[2]
print(x.shape)
net2 = DeepLabV3Decoder(in_channels=256)
net2.eval()
out = net2(x)
print(out.shape)
"""
class ResNet_FPN_Net(nn.Module):
    
    def __init__(self):
        super(ResNet_FPN_Net,self).__init__()
        self.backbone = get_encoder(
            "resnet50",
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        self.fpn = FPN(in_channels_list=[1024,512,256],out_channels=256)
        self.decoder = DeepLabV3Decoder(256)
        self.head = SegmentationHead(in_channels=256,out_channels=24,activation="sigmoid",upsampling=4)

    def forward(self,x):
        x = self.backbone(x)
        
        m,n,p = x[2],x[3],x[4]
        x =[p,n,m]
        
        x = self.fpn(x)
        x0 = F.interpolate(x[0], size=[x[2].size(2), x[2].size(3)], mode="nearest")
        x1 = F.interpolate(x[1], size=[x[2].size(2), x[2].size(3)], mode="nearest")
        x2 = x[2]
        x = x0+x1+x2
        x = self.head(x)
        return x



"""
x = torch.rand(1,3,224,224)

net = ResNet_FPN_Net()
net.eval()
y = net(x)
print(y.shape)
"""