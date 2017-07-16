# pytorch-semseg

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/pytorch-semseg/blob/master/LICENSE)

## Semantic Segmentation Algorithms Implemented in PyTorch

This repository aims at mirroring popular semantic segmentation architectures in PyTorch. 

<img src="https://meetshah1995.github.io/images/blog/ss/ptsemseg.png" width="80%">

### Networks implemented

* [Segnet](https://arxiv.org/abs/1511.00561) - With Unpooling using Maxpool indices
* [FCN](https://arxiv.org/abs/1411.4038) - All 1( FCN8s), 2 (FCN16s) and 3 (FCN8s) stream variants
* [U-Net](https://arxiv.org/abs/1505.04597) - With optional deconvolution and batchnorm
* [Link-Net](https://codeac29.github.io/projects/linknet/)

#### Upcoming 

* [PSPNet](https://arxiv.org/abs/1612.01105)
* [E-Net](https://arxiv.org/abs/1606.02147)
* [RefineNet](https://arxiv.org/abs/1611.06612)
* [MaskRCNN](https://arxiv.org/abs/1703.06870)

### DataLoaders implemented

* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)

#### Upcoming

* [NYUDv2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
* [Sun-RGBD](http://rgbd.cs.princeton.edu/)
* [MS COCO](http://mscoco.org/)

### Requirements

* pytorch >=0.1.12
* torchvision ==0.1.7
* visdom >=1.0.1 (for loss and results visualization)
* scipy
* tqdm

#### One-line installation
    
`pip install -r requirements.txt`

### Data

* Download data for desired dataset(s) from list of URLs [here](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets).
* Extract the zip / tar and modify the path appropriately in `config.json`


### Usage

Launch [visdom](https://github.com/facebookresearch/visdom#launch) by running (in a separate terminal window)

```
python -m visdom.server
```

**To train the model :**

```
python train.py [-h] [--arch [ARCH]] [--dataset [DATASET]]
                [--img_rows [IMG_ROWS]] [--img_cols [IMG_COLS]]
                [--n_epoch [N_EPOCH]] [--batch_size [BATCH_SIZE]]
                [--l_rate [L_RATE]] [--feature_scale [FEATURE_SCALE]]

  --arch           Architecture to use ['fcn8s, unet, segnet etc']
  --dataset        Dataset to use ['pascal, camvid, ade20k etc']
  --img_rows       Height of the input image
  --img_cols       Height of the input image
  --n_epoch        # of the epochs
  --batch_size     Batch Size
  --l_rate         Learning Rate
  --feature_scale  Divider for # of features to use
```

**To validate the model :**

```
python validate.py [-h] [--model_path [MODEL_PATH]] [--dataset [DATASET]]
                   [--img_rows [IMG_ROWS]] [--img_cols [IMG_COLS]]
                   [--batch_size [BATCH_SIZE]] [--split [SPLIT]]

  --model_path   Path to the saved model
  --dataset      Dataset to use ['pascal, camvid, ade20k etc']
  --img_rows     Height of the input image
  --img_cols     Height of the input image
  --batch_size   Batch Size
  --split        Split of dataset to validate on
```

**To test the model on custom images(s):**

```
python test.py [-h] [--model_path [MODEL_PATH]] [--img_path [IMG_PATH]]

  --model_path   Path to the saved model
  --img_path     Path to the input image
```