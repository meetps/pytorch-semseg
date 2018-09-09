DEBUG=True
def log(s):
    if DEBUG:
        print(s)
###################
output_channels=5
###################
import nrrd
from glob import glob
import random
import os
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
        # init_train_val_split
        ratio = 0.3
        file_paths = glob(self.root + '*_lesion*.nhdr')
        case_index = [path.split('/')[-1].split('_lesion')[0] for path in file_paths]
        case_index_UNC = [temp for temp in case_index if 'UNC' in temp]
        case_index_CHB = [temp for temp in case_index if 'CHB' in temp]
        random.shuffle(case_index_CHB)
        random.shuffle(case_index_UNC)
        val_case_index_UNC = case_index_UNC[:int(ratio * (len(case_index_UNC)))]
        val_case_index_CHB = case_index_CHB[:int(ratio * (len(case_index_CHB)))]
        val_case_index = []
        val_case_index.extend(val_case_index_CHB)
        val_case_index.extend(val_case_index_UNC)
        if self.split=='train':
            print('#####\nTrain&Split:{}\nValidationData[{}]:'.format(len(case_index), len(val_case_index)), val_case_index, '\n#####')
        # init files and annotations
        self.files = {'train':{key: [] for key in self.mods}, 'val':{key: [] for key in self.mods}}
        self.anno_files ={'train':[], 'val':[]}
        # train dataset
        for lesion_path in file_paths:
            curr_case_index = lesion_path.split('/')[-1].split('_lesion')[0]
            curr_split= 'val' if curr_case_index in val_case_index else 'train'
            for mod in self.mods:
                self.files[curr_split][mod].append(glob(self.root + curr_case_index + '*' + mod + '*' + 'nhdr')[0])
            self.anno_files[curr_split].append(lesion_path)
        if self.split=='train' and DEBUG:
            print('#####')
            print('TRAIN')
            for mod in self.mods:
                print('-{}[{}]'.format(mod, len(self.files['train'][mod])),
                      [path.split('/')[-1] for path in self.files['train'][mod]])
            print('-annot[{}]'.format(len(self.anno_files['train'])), [path.split('/')[-1] for path in self.anno_files['train']])
            print('VAL')
            for mod in self.mods:
                print('-{}[{}]'.format(mod, len(self.files['val'][mod])),
                      [path.split('/')[-1] for path in self.files['val'][mod]])
            print('-annot[{}]'.format(len(self.anno_files['val'])), [path.split('/')[-1] for path in self.anno_files['val']])
            print('#####')


    def __init__(
        self,
        root,
        split,
        is_transform=False,
        img_size=(480, 640),
        augmentations=None,
        img_norm=True,
    ):
        self.root = root
        self.is_transform = is_transform
        self.n_classes = output_channels
        self.augmentations = augmentations
        #self.img_norm = img_norm
        #self.img_size = (img_size if isinstance(img_size, tuple) else (img_size, img_size))

        self.cmap = self.color_map(normalized=False)
        self.mods = ['T1', 'T2', 'FLAIR']
        self.split = split
        self.load_data()


    def __len__(self):
        return len(self.anno_files[self.split])

    def __getitem__(self, index):
        def normalize(img):
            mini = img.min()
            return (img - mini) * 255.0 / (img.max()-mini)
        def detect_valid_area(img, loc_x, loc_y, loc_z):
            def find_start_end(vector):
                start_idx, end_idx = 0, 0
                while vector[start_idx]:
                    start_idx += 1
                while vector[end_idx-1]:
                    end_idx -= 1
                return [start_idx, end_idx]
            sum_along_12_dim_for_z = np.sum(img, (0,1)) == 0
            sum_along_23_dim_for_x = np.sum(img, (1,2)) == 0
            sum_along_13_dim_for_y = np.sum(img, (0,2)) == 0
            curr_loc_z = find_start_end(sum_along_12_dim_for_z)
            curr_loc_x = find_start_end(sum_along_23_dim_for_x)
            curr_loc_y = find_start_end(sum_along_13_dim_for_y)
            #print('curr', curr_loc_x, curr_loc_y, curr_loc_z)
            loc_x[0] = min(loc_x[0], curr_loc_x[0])
            loc_x[1] = max(loc_x[1], curr_loc_x[1])
            loc_y[0] = min(loc_y[0], curr_loc_y[0])
            loc_y[1] = max(loc_y[1], curr_loc_y[1])
            loc_z[0] = min(loc_z[0], curr_loc_z[0])
            loc_z[1] = max(loc_z[1], curr_loc_z[1])
            return loc_x, loc_y, loc_z
        img_path = {mod : self.files[self.split][mod][index] for mod in self.mods}
        lbl_path = self.anno_files[self.split][index]
        # load 4d tensor and lbl
        imgs = []
        loc_x, loc_y, loc_z = [512, -512], [512, -512], [512, -512]
        for mod in self.mods:
            img = nrrd.read(img_path[mod])[0]
            img = normalize(img)
            loc_x, loc_y, loc_z = detect_valid_area(img, loc_x, loc_y, loc_z)
            imgs.append(img)
        img = np.stack(imgs, axis = 3) # xyz * channels
        print(img.shape)
        lbl = nrrd.read(lbl_path)[0]
        lbl = np.array(lbl, dtype=np.uint8)
        img = img[loc_x[0]:loc_x[1], loc_y[0]:loc_y[1], loc_z[0]:loc_z[1], :]
        lbl = lbl[loc_x[0]:loc_x[1], loc_y[0]:loc_y[1], loc_z[0]:loc_z[1]]
        log((img.shape, lbl.shape))
        log((type(img), type(lbl)))



        #img = img.transpose(3, 0, 1, 2)

        '''
        lbl = m.imread(lbl_path)
          if not (len(img.shape) == 3 and len(lbl.shape) == 2):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        '''
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        img = img[:30,:30,:30,:]
        lbl = lbl[:30, :30, :30]
        log((img.shape, lbl.shape))
        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, :, ::-1]
        img = img.astype(np.float64)
        img = img.transpose(3, 0, 1, 2)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl))
        log((classes, np.unique(lbl)))
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

