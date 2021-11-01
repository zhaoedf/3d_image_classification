import glob
import os

from PIL import Image
from torch.utils.data import Dataset

import numpy as np 
import SimpleITK as sitk
from readseq import load_scan,resample

class SkinDataset(Dataset):
    """Dataset Caltech 256
    Class number: 257
    Train data number: 24582
    Test data number: 6027

    """
    def __init__(self):
        self.img_paths = ['./T2']
        self.targets = [1]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        seqs, reader = load_scan(self.img_paths[idx]) # T2WI
        resampled_seqs = resample(seqs)
        imgs = sitk.GetArrayFromImage(resampled_seqs)
        imgs = np.expand_dims(imgs, axis=0)

        return imgs, self.targets[idx]

    def __repr__(self):
        repr = """Caltech-256 Dataset:
        \tClass num: {}
        \tData num: {}""".format(self.class_num, self.__len__())
        return repr
