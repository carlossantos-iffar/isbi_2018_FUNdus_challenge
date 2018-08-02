import argparse

from PIL import Image

import numpy as np
import os
from scipy.misc import imresize

h_target, w_target = 2560, 2560

    
def get_mask(filename, task):
    # skip processed files
    filename = filename.replace(".jpg", "") + "_{}".format(task) + ".tif"
    if os.path.exists(filename):
        img = image2arr(filename)
        img = img[:, 230:3730, ...]
        max_len = max(img.shape[0], img.shape[1])
        img_h, img_w = img.shape
        padded = np.zeros((max_len, max_len), dtype=np.uint8)
        padded[(max_len - img_h) // 2:(max_len - img_h) // 2 + img_h, (max_len - img_w) // 2:(max_len - img_w) // 2 + img_w, ...] = 255 * img
        resized_img = imresize(padded, (h_target, w_target), 'bicubic')
        resized_img[resized_img > 0] = 1
    else:
        resized_img=np.zeros((h_target, w_target))
    return resized_img


def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def image2arr(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    return img_arr


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [fname for fname in os.listdir(path)]
        else:
            filenames = [fname for fname in os.listdir(path) if fname.endswith(extension)]
    
    if sort:
        filenames = sorted(filenames)
    
    return filenames


# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--n_processes',
    type=int,
    required=True
    )
FLAGS, _ = parser.parse_known_args()

# set paths and make dirs
home_dir = "../../data/original"
tasks = ["EX", "HE", "MA", "SE"]
task_dir_list = [os.path.join(home_dir, task) for task in tasks]
AR_dir = "../../data/original/Apparent_Retinopathy"
out_dir = "../../data/preprocessed_2560/multiclass_mask"
if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
all_AR_files = all_files_under(AR_dir)
for fname in all_AR_files:
    base_fname = os.path.basename(fname)
    multiclass_mask = np.zeros((h_target, w_target))
    for index, task_dir in enumerate(task_dir_list):
        mask = get_mask(os.path.join(task_dir, base_fname), tasks[index])
        multiclass_mask[mask == 1] = mask[mask == 1] * (index + 1)

    fname_out = os.path.join(out_dir, os.path.basename(base_fname).replace("jpg", "tif").replace("anotExpert1", "image"))
    Image.fromarray(multiclass_mask.astype(np.uint8)).save(fname_out)
    
