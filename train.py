

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
from flash.video import VideoClassificationData, VideoClassifier

# torchmetrics
import torchmetrics

from dataset import SkinDataset


# https://github.com/daili0015/ModelFeast/blob/master/tutorials/ModelZoo.md#2-3d-convolutional-neural-network
from modelfeast.models.StereoCNN import resnet18v2_3d
@ImageClassifier.backbones(name='zdf/resnet18_3d')
def fn(pretrained=False):
    # backbone =r3d_18(pretrained=pretrained)
    num_features = 2 # 二分类
    backbone = resnet18v2_3d(n_classes=num_features, in_channels=1)
    
    return backbone, num_features


# set the random seeds.
seed_everything(42)

# --------------------------------------------
# 1. prepare the data
# --------------------------------------------
train = SkinDataset()
test = SkinDataset()
# train = CIFAR10(root='/data/Public/Datasets/', train=True)
# test = CIFAR10(root='/data/Public/Datasets/', train=False)

train_transforms = T.Compose([
                # T.RandomHorizontalFlip(),
                # T.RandomRotation((0, 30)),
                T.Resize((256, 256)),
                T.RandomResizedCrop((224, 224)),
                T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225)),
            ])
test_transforms = None

'''
    .from_datasets API can not be found in doc(2021.10.08).
    
    this API can: 
        1. take any dataset class instance that inherited from torch.utils.data.Dataset as input.
        2. apply desired transform to the train/val/test set.
        3. automatically split train and val set(split from original train_dataset parameter).
        
    the example parameters below are enough for all the scenarios i might encounter.
    for more info about the params, you can view from_csv/from_data_frame API in: https://lightning-flash.readthedocs.io/en/latest/api/generated/flash.image.classification.data.ImageClassificationData.html#imageclassificationdata
'''
datamodule = VideoClassificationData.from_datasets(
    train_dataset=train,
    # val_dataset=val,
    test_dataset=train,
    # val_split = 0.1,
    batch_size = 32,
    num_workers = 8,
    train_transform = test_transforms,
    val_transform = test_transforms,
    test_transform = test_transforms
)
# datamodule = ImageClassificationData.from_folders(
    # train_folder='./data/train',
    # # val_dataset=val,
    # test_folder='./data/test',
    # val_split = 0.1,
    # batch_size = bs,
    # num_workers = 8,
    # train_transform = train_transforms,
    # val_transform = train_transforms,
    # test_transform = test_transforms
# )
# |---- train
# |----|---- cls1
# |----|----|---- XX.jpg
# |----|----|---- ...
# |----|---- cls2
# |----|---- ....
# |----|---- clsn

# --------------------------------------------
# 2. Build the model using desired Task
# --------------------------------------------
# reference link: https://lightning-flash.readthedocs.io/en/stable/api/generated/flash.image.classification.model.ImageClassifier.html#flash.image.classification.model.ImageClassifier
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

train.class_num = 2
model = VideoClassifier(
    backbone="zdf/resnet18_3d",
    pretrained=False,
    num_classes=train.class_num,
    learning_rate=std_lr,
    loss_fn=loss_func,
    optimizer=optimizer,
    optimizer_kwargs=optimizer_params,
    scheduler=lr_scheduler,
    scheduler_kwargs=scheduler_params,
    metrics=torchmetrics.Accuracy()
)
'''
for "multi_label" param, see: https://lightning-flash.readthedocs.io/en/stable/reference/image_classification_multi_label.html#image-classification-multi-label
for "serializer", see: https://lightning-flash.readthedocs.io/en/latest/api/generated/flash.core.classification.Labels.html?highlight=Labels#labels
'''

# --------------------------------------------
# 3. Create the trainer (run one epoch for demo)
# --------------------------------------------
'''
    though in lightning flash, the trainer seems like brand-new class called: "flash.core.trainer.Trainer",
    however, it is actually inherited from the trainer class in pytorch lightning(proof: "class Trainer(PlTrainer):"),
    thus, *all params(e.g. callbacks) that work for pl.trainer will also work well with the flash trainer*, since the flash trainer call "super().__init__(*args, **kwargs)".
'''
EPOCHS = 1

callbacks = [
    LearningRateMonitor(logging_interval='epoch') # or 'step'
]

trainer = flash.Trainer(
    max_epochs=EPOCHS,
    gpus='4',
    callbacks=callbacks
)

# --------------------------------------------
# 4. fit the data
# --------------------------------------------
trainer.fit(model, datamodule=datamodule)

# trainer.finetune(model, datamodule=datamodule, strategy="freeze")
'''
you can also *finetune* the model. see: 
    [API] https://lightning-flash.readthedocs.io/en/stable/api/generated/flash.core.trainer.Trainer.html#flash.core.trainer.Trainer.finetune
    [explanation of 4 default finetune strategies] https://lightning-flash.readthedocs.io/en/stable/general/finetuning.html#finetune-strategies
    [custom finetune] https://lightning-flash.readthedocs.io/en/stable/general/finetuning.html#finetuning
when using *finetune*, a strategy param is needed!
'''

# --------------------------------------------
# 5. test the model!
# --------------------------------------------
trainer.test(model, datamodule=datamodule)

# --------------------------------------------
# 6. Save the model!
# --------------------------------------------
trainer.save_checkpoint("image_classification_model.pt")

# load
# model = ImageClassifier.load_from_checkpoint("image_classification_model.pt")