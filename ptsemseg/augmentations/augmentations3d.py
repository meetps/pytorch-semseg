# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
import random
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose3d(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):

        for a in self.augmentations:
            img, mask = a(img, mask)

        return img, mask


class RandomFlip3d(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img  = np.flip(img, 0).copy()
            mask = np.flip(mask, 0).copy()
        if random.random() < 0.5:
            img = np.flip(img, 1).copy()
            mask = np.flip(mask, 1).copy()
        if random.random() < 0.5:
            img = np.flip(img, 2).copy()
            mask = np.flip(mask, 2).copy()
        return img, mask


class RandomRotate3d(object):
    def __call__(self, img, mask):
        dim1 = random.randint(0, 3)
        dim2 = random.randint(0, 3)
        dim3 = random.randint(0, 3)

        img  = np.rot90(np.rot90(np.rot90(img, dim1, (0, 1)), dim2, (0, 2)), dim3, (1, 2)).copy()
        mask  = np.rot90(np.rot90(np.rot90(mask, dim1, (0, 1)), dim2, (0, 2)), dim3, (1, 2)).copy()
        return img, mask
