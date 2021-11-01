

import os
import shutil
import SimpleITK as sitk
import cv2
import numpy as np

def load_scan(PATH):
        print("Reading Dicom directory:", PATH)
        reader = sitk.ImageSeriesReader()
        reader.LoadPrivateTagsOn()
        reader.MetaDataDictionaryArrayUpdateOn()

        dicom_names = reader.GetGDCMSeriesFileNames(PATH)
        reader.SetFileNames(dicom_names)

        image = reader.Execute()

        size = image.GetSize()
        print("Image size:", size[0], size[1], size[2])
        
        return image,reader #type(image) = SimpleITK.SimpleITK.Image


def resample(image, new_spacing=[1,1,1],new_size=[-1,-1,-1]):
        resampler = sitk.ResampleImageFilter()

        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        '''
        sitkNearestNeighbor = 1,
        sitkLinear = 2,
        sitkBSpline = 3,
        sitkGaussian = 4,
        sitkLabelGaussian = 5
        '''
        old_size = image.GetSize()
        if new_size == [-1,-1,-1]:
                old_spacing = image.GetSpacing()
        #print(spacing)
                new_size = [int(old_size[i] * float(old_spacing[i]) / new_spacing[i]) for i in range(3)]
        #这个计算公式没有问题！见OneNote笔记易懂 checked

        resampler.SetSize(new_size)  # mandatory

        new_image = resampler.Execute(image)
        new_size = new_image.GetSize()
        print("new Image size:", new_size[0], new_size[1], new_size[2])

        return new_image

# seqs, reader = load_scan('/data/xuchen/post-A15/DWI') # T2WI
# # seqs, reader = load_scan('./PATIENT_DICOM')
# resampled_seqs = resample(seqs)
# # np.array(seqs)
# imgs = sitk.GetArrayFromImage(resampled_seqs)
# imgs = np.expand_dims(imgs, axis=0)
# print(imgs.shape)

# # os.mkdir('./xuchen')
# for idx, img in enumerate(imgs):
#         img = np.expand_dims(img, axis=2)
#         new_img = img.copy()
#         print(new_img.max())
#         # new1_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
#         # print(np.unique(img))
#         cv2.imwrite(f'./xuchen/{idx}.png', new_img)