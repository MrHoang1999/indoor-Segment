import matplotlib.pyplot as plt

from lib import *
from predict_function import *
from dataset import *
model = torch.load("D:\\naver\\Unet-Mobilenet_v2_mIoU-0.977.pt")
VOC_COLORMAP = [[0, 0, 0], [255,0,0]]

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

image2, mask2 = test_set[34]
pred_mask2, score2 = predict_image_mask_miou(model, image2, mask2)
x = colour_code_segmentation(pred_mask2.numpy(),VOC_COLORMAP)
print(x.shape)
yz = (np.array(image2))*0.35 + x*0.65

yz = torch.from_numpy(yz).int()
yz= yz.numpy()

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
ax1.imshow(image2)
ax1.set_title('Picture');

ax2.imshow(mask2)
ax2.set_title('Ground truth')
ax2.set_axis_off()

ax3.imshow(yz)
ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score2))
ax3.set_axis_off()
plt.show()