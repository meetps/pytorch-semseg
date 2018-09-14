DEBUG=False
def log(s):
    if DEBUG:
        print(s)
###################
from scipy.ndimage.interpolation import zoom as im_zoom
import nrrd
from glob import glob
import random
import os
import time
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt

from torch.utils import data
from glob import glob
from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *


class miccai2008Loader(data.Dataset):
    def load_data(self):
        if self.split=='train':
            print('#####\nTrain&Split:{}\nValidationData[{}]:'.format(len(self.split_info['case_index']),
                                                                      len(self.split_info['val_case_index'])),
                                            self.split_info['val_case_index'], '\n#####')
        # init files and annotations
        self.files = {'train':{key: [] for key in self.mods}, 'val':{key: [] for key in self.mods}}
        self.anno_files ={'train':[], 'val':[]}
        for lesion_path in self.split_info['file_paths']:
            curr_case_index = lesion_path.split('/')[-1].split('_lesion')[0]
            curr_split= 'val' if curr_case_index in self.split_info['val_case_index'] else 'train'
            for mod in self.mods:
                self.files[curr_split][mod].append(glob(self.root + curr_case_index + '*' + mod + '*' + 'nhdr')[0])
            self.anno_files[curr_split].append(lesion_path)
        if self.split=='train':
            if False:
                print('#####')
                print('TRAIN')
                for mod in self.mods:
                    print('-{}[{}]'.format(mod, len(self.files['train'][mod])),
                          [path.split('/')[-1].replace('_train_','_').split('.')[0] for path in self.files['train'][mod]])
                print('-annot[{}]'.format(len(self.anno_files['train'])), [path.split('/')[-1].replace('_train_','_').split('.')[0] for path in self.anno_files['train']])
                print('VAL')
                for mod in self.mods:
                    print('-{}[{}]'.format(mod, len(self.files['val'][mod])),
                          [path.split('/')[-1].replace('_train_','_').split('.')[0] for path in self.files['val'][mod]])
                print('-annot[{}]'.format(len(self.anno_files['val'])), [path.split('/')[-1].replace('_train_','_').split('.')[0] for path in self.anno_files['val']])

    def __init__(
        self,
        root,
        split,
        is_transform=False,
        img_size=(480, 640),
        augmentations=None,
        img_norm=True,
        split_info=None,
        patch_size=None
    ):
        self.root = root
        self.is_transform = is_transform
        self.n_classes = 2
        self.augmentations = augmentations
        self.split_info = split_info
        self.patch_size = 512 if patch_size is None else patch_size
        #self.lbl_unet_outputs_size = self.patch_size
        #self.img_norm = img_norm
        #self.img_size = (img_size if isinstance(img_size, tuple) else (img_size, img_size))

        self.cmap = self.color_map(normalized=False)
        self.mods = ['T1', 'T2', 'FLAIR']
        self.split = split
        self.load_data()
        self.anno_files[self.split] = self.anno_files[self.split]#[:2] ## DEBUG
    def __len__(self):
        return len(self.anno_files[self.split])
    def normalize(self, img):
        return (img - img.min()) / (img.max() - img.min())
    def randomCrop3D(self, img, lbl):
        while True:
            x = random.randint(0, img.shape[0] - self.patch_size)
            y = random.randint(0, img.shape[1] - self.patch_size)
            z = random.randint(0, img.shape[2] - self.patch_size)
            lbl_cropped = lbl[x:x + self.patch_size, y:y + self.patch_size, z:z + self.patch_size]
            if lbl_cropped.sum() > 0:
                img_cropped = img[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size, :]
                return img_cropped, lbl_cropped
    def __getitem__(self, index):
        st = time.time()
        img_path = {mod : self.files[self.split][mod][index] for mod in self.mods}
        lbl_path = self.anno_files[self.split][index]
        case_idx = lbl_path.split('/')[-1].split('_lesion')[0]
        # load 4d tensor and lbl
        imgs = []
        for mod in self.mods:
            img = nrrd.read(img_path[mod])[0]
            img = self.normalize(img)
            imgs.append(img)
        img = np.stack(imgs, axis = 3) # xyz * channels
        lbl = nrrd.read(lbl_path)[0]
        lbl = np.array(lbl, dtype=np.uint8)

        # RandomCrop to patchsize
        img, lbl = self.randomCrop3D(img, lbl)
        log((lbl_path, img.shape, lbl.shape))

        # augmentation specified in [yml]
        log(('self.augmentations', self.augmentations))
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        # transform   #input: xyzc => output:cxyz
        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        log((img.shape, lbl.shape))
        log((index, self.split, lbl_path, 'loadingTime:{}'.format(time.time()-st)))
        return img, lbl, case_idx

    def transform(self, img, lbl):
        #img = img[:, :, :, ::-1]
        img = img.astype(np.float64)
        img = img.transpose(3, 0, 1, 2)

        '''
        # Resize label map for UNET  [124=>36]
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl =im_zoom(lbl, zoom=self.lbl_unet_outputs_size/self.patch_size, mode="constant")
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl)), (classes, np.unique(lbl))
        '''

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def color_map(self, N=256, normalized=False):
        """
        Return Color Map in PASCAL VOC format
        """

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255.0 if normalized else cmap
        return cmap

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.cmap[l, 0]
            g[temp == l] = self.cmap[l, 1]
            b[temp == l] = self.cmap[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

