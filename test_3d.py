
import torch

from torchvision.models.video import r3d_18

from modelfeast.models.StereoCNN import resnet18v2_3d


# model = r3d_18(pretrained=True)
model = resnet18v2_3d(n_classes=10, in_channels=1)

x = torch.randn(3,1,16,112,112)

y = model(x)

print(y.shape)