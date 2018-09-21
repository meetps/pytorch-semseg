# normalization and padding
from glob import glob
import nibabel as nib
from tqdm import tqdm
import numpy as np
path = "/media/machine/Storage/Dataset/BrainLesion/sasha/3D.nii.gz/*/*/*.nii.gz"
paths = glob(path)
padding_to = 224
root= '/media/machine/Storage/Dataset/BrainLesion/sashaSplit/train&val/'
def normalize(img):
    return (img - img.min()) / (img.max() - img.min())
for path in tqdm(paths):
    tokens = path.split('/')
    case_idx = tokens[-1].split('.')[0]
    case_mod = tokens[-2]
    new_filename = case_idx + '-' + case_mod.upper() + '_preprocessed.npy'
    new_path = root + new_filename
    data = nib.load(path).get_data()
    data_norm = normalize(data)
    dim1 = (padding_to - data.shape[0]) // 2
    dim2 = (padding_to - data.shape[1]) // 2
    dim3 = (padding_to - data.shape[2]) // 2
    data_pad = np.pad(data_norm, ((dim1, dim1 + 1), (dim2, dim2 + 1), (dim3, dim3 + 1)), mode='constant')
    np.save(new_path, data_pad)