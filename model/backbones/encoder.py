from segmentation_models_pytorch.encoders import get_encoder
import torch

encoder = get_encoder(
            "resnet50",
            in_channels=3,
            depth=5,
            weights="imagenet",
          
        )

x = torch.rand(1,3,224,224)
y = encoder(x)
y = y[2:5]
y =reversed(y)
for i,xt in enumerate(y):
    print(i)
    print(xt.shape)