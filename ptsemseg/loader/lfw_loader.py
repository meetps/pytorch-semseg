import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from tqdm import tqdm
from torch.utils import data

def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    js = open('config.json').read()
    data = json.loads(js)
    return os.path.expanduser(data[name]['data_path'])

class LFWLoader(data.Dataset):

    def __init__(self, root, split='train_aug', is_transform=False,
                 img_size=250, augmentations=None, n_streams=1):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 3
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
        else (img_size, img_size)
        for split in ['train', 'validation', 'test']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [(id_.rstrip('\n'), 0) for id_ in file_list]
            if n_streams==2:
                path = pjoin(self.root, 'Mugsy', split + '.txt')
                file_list_mugsy = tuple(open(path, 'r'))
                file_list_mugsy = [(id_.rstrip('\n'), 1) for id_ in file_list_mugsy]
                file_list += file_list_mugsy
            print("in loading..")
            print(split, len(file_list))
            self.files[split] = file_list
        # self.setup_annotations()

    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, index):
        name_idx, ds = self.files[self.split][index]
        # print("Get_item:", ds)
        # if ds == 0:
        #     name, idx = name_idx.split()
        #     im_path = pjoin(self.root, "images", name, name + "_" + idx.zfill(4) + ".jpg")
        #     lbl_path = pjoin(self.root, "masks", name + "_" + idx.zfill(4) + ".ppm")
        # else:
        #     camera, name = name_idx.split()
        #     print(camera, name)
        #     im_path = pjoin(self.root, "Mugsy/images/images_unnested", "cam" + camera + "_image" + name + ".png")
        #     lbl_path = pjoin(self.root, "Mugsy/masks/masks", camera + ".png")

        # im = m.imread(im_path)
        # im = np.array(im, dtype=np.uint8)
        # lbl = m.imread(lbl_path)
        # lbl = np.array(lbl, dtype=np.int8)

        # if ds == 0:
        #     lbl = self.encode_segmap(lbl)
        # else:
        #     print("here")
        #     lbl = self.encode_mugsy_segmap(lbl)
        #     print(lbl[:,125])
        #     print(sum(lbl[:,125]))

        # if self.augmentations is not None:
        #     im, lbl = self.augmentations(im, lbl)
        # if self.is_transform:
        #     im, lbl = self.transform(im, lbl)
        # print(ds)
        # return im, lbl, ds
        # print("Get_item:", ds)
        if ds == 0:
            name, idx = name_idx.split()
            im_path = pjoin(self.root, "images", name, name + "_" + idx.zfill(4) + ".jpg")
            lbl_path = pjoin(self.root, "masks", name + "_" + idx.zfill(4) + ".ppm")
            im = m.imread(im_path)
            im = np.array(im, dtype=np.uint8)
            lbl = m.imread(lbl_path)
            lbl = np.array(lbl, dtype=np.int8)
            lbl = self.encode_segmap(lbl)
            if self.augmentations is not None:
                im, lbl = self.augmentations(im, lbl)
            if self.is_transform:
                im, lbl = self.transform(im, lbl)
            return im, lbl, 0
        else:
            camera, name = name_idx.split()
            # print(camera, name)
            im_path = pjoin(self.root, "Mugsy/images/images_unnested", "cam" + camera + "_image" + name + ".png")
            lbl_path = pjoin(self.root, "Mugsy/masks/masks", camera + ".png")
            im = m.imread(im_path)
            im = np.array(im, dtype=np.uint8)
            lbl = m.imread(lbl_path)
            lbl = np.array(lbl, dtype=np.int8)
            # print("here")
            lbl = self.encode_mugsy_segmap(lbl)
            # print(lbl[:,125])
            # print(sum(lbl[:,125]))
            if self.augmentations is not None:
                im, lbl = self.augmentations(im, lbl)
            if self.is_transform:
                im, lbl = self.transform(im, lbl)
            return im, lbl, 1


    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        lbl[lbl==255] = 0
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest',
                         mode='F')
        lbl = lbl.astype(int)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (3, 3)
        """
        return np.asarray([[0,0,-1], [0,-1,0], [-1,0,0]])

    def get_mugsy_labels(self):
        return np.asarray([[0,0,0], [-1,-1,-1]])


    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def encode_mugsy_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_mugsy_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r * -255.0
        rgb[:, :, 1] = g * -255.0
        rgb[:, :, 2] = b * -255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


    def decode_mugsy_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_mugsy_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, len(label_colours)):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r * -255.0
        rgb[:, :, 1] = g * -255.0
        rgb[:, :, 2] = b * -255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb





