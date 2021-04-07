# DATASET

import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2

from PIL import Image
from tqdm import tqdm
from torch.utils import data as dt
from torchvision import transforms
import warnings

class pascalVOCLoader(dt.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of four data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        test: an arbitrary set of images from the "val" set, written in a text file test.txt in the same directory
    """

    def __init__(
        self,
        root="/gpfs/home/fnouraei/data/fnouraei/VOC/VOCdevkit/VOC2012/", # choose your main path as default
        sbd_path="/gpfs/home/fnouraei/data/fnouraei/VOC/benchmark_RELEASE", # choose your SBD dir as default
        split="train", # loader split {"train", "val", "test", "train_aug"}
        is_transform=True,
        img_size=(128,128), # choose default input size
        augmentations=None,
        img_norm=True,
        aug_with_sbd = False, # choose whether or not to add Berkeley images to val and train sets by default
    ):
        self.root = root
        self.sbd_path = sbd_path
        self.aug_with_sbd = aug_with_sbd
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 21
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        
        for split in ["train", "val", "test"]:
            path = pjoin(self.root, "ImageSets","Segmentation", split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        self.setup_annotations()
        
        if not self.img_norm:
            self.tf = transforms.ToTensor()
        else: 
            self.tf = transforms.Compose(
                [
                
                    transforms.ToTensor()
                    ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])



    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded", im_name + ".png")
        if self.aug_with_sbd: 
            lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded_aug", im_name + ".png")
        im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(lbl_path,  cv2.IMREAD_UNCHANGED)
        

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        
        return im, lbl

    def transform(self, img, lbl):
        if self.img_size == ("same", "same"):
            pass
        else:
            img = cv2.resize(img, (self.img_size[0], self.img_size[1]), interpolation =cv2.INTER_NEAREST)
            lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation =cv2.INTER_NEAREST)
        
        
        img = self.tf(img)
    
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = 0
        return img, lbl

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray( [
                [0, 0, 0],        #0: bg
                [128, 0, 0],      #1: aeroplane
                [0, 128, 0],      #2: bicycle   
                [128, 128, 0],    #3: bird  
                [0, 0, 128],      #4: boat  
                [128, 0, 128],    #5: bottle 
                [0, 128, 128],    #6: bus 
                [128, 128, 128],  #7: car 
                [64, 0, 0],       #8: cat
                [192, 0, 0],      #9: chair
                [64, 128, 0],     #10: cow
                [192, 128, 0],    #11: dining table
                [64, 0, 128],     #12: dog
                [192, 0, 128],    #13: horse
                [64, 128, 128],   #14: motorbike 
                [192, 128, 128],  #15: person
                [0, 64, 0],       #16: potted plant
                [128, 64, 0],     #17: sheep
                [0, 192, 0],      #18: sofa
                [128, 192, 0],    #19: train
                [0, 64, 128],     #20: tv/monitor
            ] )
    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
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
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup_annotations(self):
        """
        pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). 
        """
        sbd_path = self.sbd_path
        if not self.aug_with_sbd:
            target_path = pjoin(self.root, "SegmentationClass/pre_encoded")
        if self.aug_with_sbd:
            target_path = pjoin(self.root, "SegmentationClass/pre_encoded_aug")
            
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        path = pjoin(sbd_path, "dataset", "train.txt")
        sbd_train_list = tuple(open(path, "r"))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
        
        if not self.aug_with_sbd:
            train = self.files["train"]
            
        if self.aug_with_sbd:
            train_aug = self.files["train"] + sbd_train_list
        
        val = self.files["val"]            
        test = self.files["test"]
        
        if self.aug_with_sbd:
            # keep unique elements (stable)
            train_aug = [train_aug[i] for i in sorted(np.unique(train_aug, return_index=True)[1])]
            self.files["train_aug"] = train_aug
        
        pre_encoded = glob.glob(pjoin(target_path, "*.png"))
        
        if not self.aug_with_sbd:
            expected = len(list(set().union(val,test,train)))
            # make sure validation set is held out
            #print("[Debug] val and train intersection: ", len(list(set().intersection(val,train))))
        else:
            expected = len(list(set().union(val,test,train_aug)))
            # make sure validation set is held out
            #print("[Debug] val and train intersection: ", len(list(set().intersection(val,train_aug))))
            
        
 
        if len(pre_encoded) < expected: 
            
            if self.aug_with_sbd:
                for ii in tqdm(sbd_train_list, desc="pre-encode SBD"):
                    fname = ii + ".png"
                    lbl_path = pjoin(sbd_path, "dataset/cls", ii + ".mat")
                    data = io.loadmat(lbl_path)
                    lbl = data["GTcls"][0]["Segmentation"][0].astype(np.int32)
                    cv2.imwrite(pjoin(target_path, fname), lbl)
            
            for split in ["train", "val", "test"]:
            
                for ii in tqdm(self.files[split], desc="pre-encode "+split):
                    fname = ii + ".png"
                    lbl_path = pjoin(self.root, "SegmentationClass", fname)
                    lbl = cv2.imread(lbl_path,  cv2.IMREAD_UNCHANGED)
                    lbl = cv2.cvtColor(np.array(lbl), cv2.COLOR_BGR2RGB) 
                    lbl = self.encode_segmap(lbl)
                    #print("[Debug] pre-encoded label before saving as png: ", np.unique(lbl))
                    cv2.imwrite(pjoin(target_path, fname), lbl)
        
        pre_encoded = glob.glob(pjoin(target_path, "*.png"))            
        print("[info] num expected labels: {}  num pre-encoded labels: {}".format(expected,len(pre_encoded)))            

        
"""        
# Test Pascal Dataloader (JUPYTER NOTEBOOK)

%matplotlib inline
import matplotlib.pyplot as plt
from ptsemseg.augmentations.augmentations import (
    AdjustContrast,
    AdjustGamma,
    AdjustBrightness,
    AdjustSaturation,
    AdjustHue,
    RandomCrop,
    RandomHorizontallyFlip,
    RandomVerticallyFlip,
    Scale,
    RandomSized,
    RandomSizedCrop,
    RandomRotate,
    RandomTranslate,
    CenterCrop,
    Compose,
)

if __name__ == '__main__':
    
    local_path = '/gpfs/home/fnouraei/data/fnouraei/VOC/VOCdevkit/VOC2012/'
    bs = BATCH_SIZE
    if OVERFIT:
        dset = pascalVOCLoader_reduced(root=local_path, split = 'train',img_size=(INPUT_SIZE,INPUT_SIZE)
                               , is_transform=True , aug_with_sbd=False
                               , augmentations= None)
    else:
        dset = pascalVOCLoader(root=local_path, split = 'train',img_size=(INPUT_SIZE,INPUT_SIZE)
                               , is_transform=True , aug_with_sbd=True
                               , augmentations= Compose([RandomRotate(10), RandomHorizontallyFlip(p=0.5)
                                                          ,RandomVerticallyFlip(p=0.5),RandomSizedCrop(125)
                                                          ,AdjustGamma(gamma = 2.2),AdjustContrast(cf=0.4)]))

                                                      
    print("[Debug] length of dataset: ",len(dset))
    
    trainloader = dt.DataLoader(dset, batch_size=bs, shuffle=True)

    for i, data in enumerate(trainloader): # i = batch idx , data = (img , label) (as tensors - first dim is batch size)
        img, label = data
        img = img.numpy()
        img = np.transpose(img, [0,2,3,1])
        print("[Debug] label values:",label.unique())
    
        f, axarr = plt.subplots(bs, 2, figsize=(15, 15), dpi=80)
        
        for j in range(bs):
            print("[Debug] batch item {} img shape: {} label shape: {}".format(bs,img[j].shape,label.numpy()[j].shape))
            axarr[j,0].imshow(img[j])
            axarr[j,1].imshow(dset.decode_segmap((label[j].numpy())))
        plt.show()
    
    plt.close()
"""        
        
        
        
        
        
 
