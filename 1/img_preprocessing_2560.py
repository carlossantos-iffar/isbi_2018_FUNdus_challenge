import argparse

from PIL import Image

import numpy as np
import os
import multiprocessing
from scipy.misc import imresize

h_target, w_target = 2560, 2560

    
def process(args):
    filenames, out_dir = args
    for filename in filenames:
        # skip processed files
        out_file = os.path.join(out_dir, os.path.basename(filename).replace("jpg", "tif").replace("anotExpert1", "image"))
        if os.path.exists(out_file):
            continue

        img = image2arr(filename)
        img = img[:, 230:3730, ...]
        # pad imgs
        max_len = max(img.shape[0], img.shape[1])
        if len(img.shape) == 3:  # rgb image
            img_h, img_w, _ = img.shape
            padded = np.zeros((max_len, max_len, 3), dtype=np.uint8)
            padded[(max_len - img_h) // 2:(max_len - img_h) // 2 + img_h, (max_len - img_w) // 2:(max_len - img_w) // 2 + img_w, ...] = img
            resized_img = imresize(padded, (h_target, w_target), 'bicubic')
        else:  # segmentation label
            img_h, img_w = img.shape
            padded = np.zeros((max_len, max_len), dtype=np.uint8)
            padded[(max_len - img_h) // 2:(max_len - img_h) // 2 + img_h, (max_len - img_w) // 2:(max_len - img_w) // 2 + img_w, ...] = 255 * img
            resized_img = imresize(padded, (h_target, w_target), 'bicubic')
            resized_img[resized_img > 0] = 1
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
FLAGS, _ = parser.parse_known_args()

# set paths and make dirs
home_dir = "../../data/original"
sub_dirs = ["Apparent_Retinopathy", "EX", "HE", "MA", "No_Apparent_Retinopathy", "SE"]
dir_list = [os.path.join(home_dir, sub_dir) for sub_dir in sub_dirs]

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
    out_dir = dirname.replace("original", "preprocessed_2560")
    if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    pool = multiprocessing.Pool(processes=FLAGS.n_processes)
    pool.map(process, [(filenames[i], out_dir) for i in range(FLAGS.n_processes)])
