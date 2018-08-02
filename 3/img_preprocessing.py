import argparse

from PIL import Image

import numpy as np
import os
import multiprocessing
from scipy.misc import imresize

h_target, w_target = 640, 640

    
def process(args):
    filenames, out_dir = args
    for filename in filenames:
        # skip processed files
        out_file = os.path.join(out_dir, os.path.basename(filename).replace("jpg", "tif").replace("anotExpert1", "image"))
        if os.path.exists(out_file):
            continue

        img = image2arr(filename)
        if "image_" in os.path.basename(out_file):  # crop image for drion (to exclude marking)
            img = img[:, 50:, ...]
        elif "IDRiD_" in os.path.basename(out_file):
            img = img[:, 230:3730, ...]
        # pad imgs
        max_len = max(img.shape[0], img.shape[1])
        if len(img.shape) == 3:  # rgb image
            img_h, img_w, _ = img.shape
            padded = np.zeros((max_len, max_len, 3))
        else:
            img_h, img_w = img.shape
            padded = np.zeros((max_len, max_len))
        padded[(max_len - img_h) // 2:(max_len - img_h) // 2 + img_h, (max_len - img_w) // 2:(max_len - img_w) // 2 + img_w, ...] = img
        resized_img = imresize(padded, (h_target, w_target), 'bilinear')
        Image.fromarray(resized_img).save(out_file)
          

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
parser.add_argument(
    '--task',
    type=str,
    required=True
    )
FLAGS, _ = parser.parse_known_args()

# set paths and make dirs

if FLAGS.task == "segmentation":
    in_train_dir = "../data/disc_segmentation/training_before_preprocessing"
    in_val_dir = "../data/disc_segmentation/validation_before_preprocessing"
    in_train_img_dir = os.path.join(in_train_dir, "images")
    in_train_mask_dir = os.path.join(in_train_dir, "masks")
    in_val_img_dir = os.path.join(in_val_dir, "images")
    in_val_mask_dir = os.path.join(in_val_dir, "masks")
    dir_list = [in_train_img_dir, in_train_mask_dir, in_val_img_dir, in_val_mask_dir]
if FLAGS.task == "localization":
    in_train_dir = "../data/localization/training_before_preprocessing"
    in_val_dir = "../data/localization/validation_before_preprocessing"
    dir_list = [in_train_dir, in_val_dir]

# run multi-process
for dirname in dir_list:
    all_files = all_files_under(dirname)
    filenames = [[] for __ in xrange(FLAGS.n_processes)]
    chunk_sizes = len(all_files) // FLAGS.n_processes
    for index in xrange(FLAGS.n_processes):
        if index == FLAGS.n_processes - 1:  # allocate ranges (last GPU takes remainders)
            start, end = index * chunk_sizes, len(all_files)
        else:
            start, end = index * chunk_sizes, (index + 1) * chunk_sizes
        filenames[index] = all_files[start:end]
    
    # run multiple processes
    out_dir = dirname.replace("_before_preprocessing", "")
    if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    pool = multiprocessing.Pool(processes=FLAGS.n_processes)
    pool.map(process, [(filenames[i], out_dir) for i in range(FLAGS.n_processes)])
