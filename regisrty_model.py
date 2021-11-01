
# https://pytorch.org/vision/stable/models.html?highlight=3d#torchvision.models.video.r3d_18
# from torchvision.models.video import r3d_18 # resnet18-3d

from modelfeast.models.StereoCNN import resnet18v2_3d

from functools import partial

from flash import Task
from flash.core.registry import FlashRegistry

class StereoImageClassifier(Task):

    backbones = FlashRegistry("3d_backbones")

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ):

        self.backbone, self.num_features = self.backbones.get(backbone)(pretrained=pretrained)

'''
    核心在于“FlashRegistry("backbones")”，只要它存在就行了，所以定不定义class没区别。**但是定义了可以更方便使用装饰器
    从上面的__init__函数可以看出，其实定义了class，是classifier的class，它根据参数从registry的backbone中选出了“特定”的模型。 nm
'''



@StereoImageClassifier.backbones(name='zdf/resnet18_3d')
def fn(pretrained=True):
    # backbone =r3d_18(pretrained=pretrained)
    num_features = 2 # 二分类
    backbone = resnet18v2_3d(n_classes=num_features, in_channels=1)

    return backbone, num_features

print(StereoImageClassifier.available_backbones())

# torch
import torch
from torch.utils.data import Dataset, random_split

# torchvision
# from torchvision.datasets import Caltech256
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

# pytorch lightning
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor


# lightning flash
import flash
from flash.core.classification import Labels
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier

# torchmetrics
import torchmetrics

# from dataset import Caltech256

std_lr = 1e-3
optimizer = torch.optim.SGD
optimizer_params = {
    'momentum': 0.9,
    'weight_decay': 5e-4
}

lr_scheduler = torch.optim.lr_scheduler.StepLR
scheduler_params = {
    'step_size': 2,
    'gamma': 0.1
}

loss_func = torch.nn.functional.cross_entropy


@ImageClassifier.backbones(name='zdf/resnet18_3d')
def fn(pretrained=True):
    # backbone =r3d_18(pretrained=pretrained)
    num_features = 2 # 二分类
    backbone = resnet18v2_3d(n_classes=num_features, in_channels=1)

    return backbone, num_features

model = ImageClassifier(
    backbone="zdf/resnet18_3d",
    # pretrained=True,
    num_classes=2,
    learning_rate=std_lr,
    loss_fn=loss_func,
    optimizer=optimizer,
    optimizer_kwargs=optimizer_params,
    scheduler=lr_scheduler,
    scheduler_kwargs=scheduler_params,
    metrics=torchmetrics.Accuracy()
)

x = torch.randn(3,1,16,112,112)

y = model(x)

print('1', y.shape)
