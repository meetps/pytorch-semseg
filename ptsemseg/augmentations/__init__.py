import logging
from ptsemseg.augmentations.augmentations2d import *
from ptsemseg.augmentations.augmentations3d import *

logger = logging.getLogger('ptsemseg')

key2aug2d = {'gamma': AdjustGamma,
           'hue': AdjustHue,
           'brightness': AdjustBrightness,
           'saturation': AdjustSaturation,
           'contrast': AdjustContrast,
           'rcrop': RandomCrop,
           'hflip': RandomHorizontallyFlip,
           'vflip': RandomVerticallyFlip,
           'scale': Scale,
           'rsize': RandomSized,
           'rsizecrop': RandomSizedCrop,
           'rotate': RandomRotate,
           'translate': RandomTranslate,
           'ccrop': CenterCrop,}
key2aug3d = {'flip3d': RandomFlip3d,
             'rotate3d': RandomRotate3d,
           }

def get_composed_augmentations2d(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug2d[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose2d(augmentations)
def get_composed_augmentations3d(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key in aug_dict:
        augmentations.append(key2aug3d[aug_key]())
        logger.info("Using {} aug.".format(aug_key))
    return Compose3d(augmentations)


