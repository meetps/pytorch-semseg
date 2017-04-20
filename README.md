# pytorch-semseg

[![Build Status](https://travis-ci.com/meetshah1995/pytorch-semseg.svg?token=H8ye8rbTHySsWieqJyz6&branch=master)](https://travis-ci.com/meetshah1995/pytorch-semseg)

## Semantic Segmentation Algorithms Implemented in PyTorch

This repository aims at mirroring the semantic segmentation architectures as mentioned in respective papers. 

### Networks implemented

* [Segnet](https://arxiv.org/abs/1511.00561) - With Unpooling using Maxpool indices
* [FCN](https://arxiv.org/abs/1411.4038) - All 1( FCN8s), 2 (FCN16s) and 3 (FCN8s) stream variants
* [U-Net](https://arxiv.org/abs/1505.04597) - With optional deconvolution and batchnorm

### DataLoaders implemented

* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)

### TODO
* Add dataloader for MS COCO
* Implement MaskRCNN
