import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from ptsemseg.utils import recursive_glob


class MITSceneParsingBenchmarkLoader(data.Dataset):
    """MITSceneParsingBenchmarkLoader

    http://sceneparsing.csail.mit.edu/

    Data is derived from ADE20k, and can be downloaded from here:
    http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

    NOTE: this loader is not designed to work with the original ADE20k dataset;
    for that you will need the ADE20kLoader

    This class can also be extended to load data for places challenge:
    https://github.com/CSAILVision/placeschallenge/tree/master/sceneparsing

    """

    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 151  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = {}

        self.images_base = os.path.join(self.root, "images", self.split)
        self.annotations_base = os.path.join(self.root, "annotations", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path)[:-4] + ".png")

        img = m.imread(img_path, mode="RGB")
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        if self.img_size == ("same", "same"):
            pass
        else:
            img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        if self.img_size == ("same", "same"):
            pass
        else:
            lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl) < self.n_classes):
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
