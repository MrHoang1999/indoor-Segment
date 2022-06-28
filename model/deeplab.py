import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones.resnet import resnet50
from model.decoder.deeplabv3 import *
from model.head.segment_head import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.model import SegmentationModel
class Deeplab(SegmentationModel):
    def __init__(self):
        super(Deeplab,self).__init__()
        self.backbone = get_encoder(
            "resnet50",
            in_channels=3,
            depth=5,
            weights="imagenet",
          
        )#resnet50(pretrained=True)
        self.decoder = DeepLabV3Decoder(512)
        self.head = SegmentationHead(activation="sigmoid",in_channels=256,out_channels=24,upsampling=8)
    def forward(self,x):
        x = self.backbone(x)[3]
        x = self.decoder(x)
        x = self.head(x)
        return x


