"""
crop a blob in the center of fundus (eye) images  

- input folder is given and all images are processed
- a blob with any side less than threshold length (min_height_ratio*height) will not be processed 
"""
import argparse
import os
import multiprocessing
from scipy.misc import imresize

from PIL import Image
from skimage import measure

import numpy as np

h_target, w_target = 512, 512


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


def process(args):
    filenames, out_dir = args
    for filename in filenames:
        # skip if output file exists already
        out_file = os.path.join(out_dir, os.path.basename(filename))
        if os.path.exists(out_file):
            print ("skip %s [%s already exists]" % (filename, out_file))
            continue
        
        # read the image and resize 
        img = np.array(Image.open(filename))
        h_ori, w_ori, _ = np.shape(img)
        red_threshold = 20
        roi_check_len = h_ori // 5
        
        # Find Connected Components with intensity above the threshold
        blobs_labels, n_blobs = measure.label(img[:, :, 0] > red_threshold, return_num=True)
        if n_blobs == 0:
            print ("crop failed for %s " % (filename)
                    + "[no blob found]")
            continue
       
        # Find the Index of Connected Components of the Fundus Area (the central area)
        majority_vote = np.argmax(np.bincount(blobs_labels[h_ori // 2 - roi_check_len // 2:h_ori // 2 + roi_check_len // 2,
                                                                w_ori // 2 - roi_check_len // 2:w_ori // 2 + roi_check_len // 2].flatten()))
        if majority_vote == 0:
            print ("crop failed for %s " % (filename)
                    + "[invisible areas (intensity in red channel less than " + str(red_threshold) + ") are dominant in the center]")
            continue
        
        row_inds, col_inds = np.where(blobs_labels == majority_vote)
        row_max = np.max(row_inds)
        row_min = np.min(row_inds)
        col_max = np.max(col_inds)
        col_min = np.min(col_inds)
        if row_max - row_min < 100 or col_max - col_min < 100:
            print n_blobs
            for i in range(1, n_blobs + 1):
                print len(blobs_labels[blobs_labels == i])
            print ("crop failed for %s " % (filename)
                    + "[too small areas]")
            continue
        
        # crop the image 
        crop_img = img[row_min:row_max, col_min:col_max]
        max_len = max(crop_img.shape[0], crop_img.shape[1])
        img_h, img_w, _ = crop_img.shape
        padded = np.zeros((max_len, max_len, 3), dtype=np.uint8)
        padded[(max_len - img_h) // 2:(max_len - img_h) // 2 + img_h, (max_len - img_w) // 2:(max_len - img_w) // 2 + img_w, ...] = padded[(max_len - img_h) // 2:(max_len - img_h) // 2 + img_h, (max_len - img_w) // 2:(max_len - img_w) // 2 + img_w, ...] = crop_img
        
        resized_img = imresize(crop_img, (h_target, w_target), 'bicubic')
        
        Image.fromarray(resized_img).save(out_file.replace("jpeg", "png"))


def crop_center_blob(in_dir, out_dir, n_processes):
                
    # make out_dir if not exists
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # divide files
    all_files = all_files_under(in_dir, extension='.jpeg')
    filenames = [[] for __ in xrange(n_processes)]
    chunk_sizes = len(all_files) // n_processes
    for index in xrange(n_processes):
        if index == n_processes - 1:  # allocate ranges (last GPU takes remainders)
            start, end = index * chunk_sizes, len(all_files)
        else:
            start, end = index * chunk_sizes, (index + 1) * chunk_sizes
        filenames[index] = all_files[start:end]
    
    # run multiple processes
    pool = multiprocessing.Pool(processes=FLAGS.n_processes)
    pool.map(process, [(filenames[i], out_dir) for i in range(FLAGS.n_processes)])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dir',
        type=str,
        help="Directory of input image files",
        required=True
        )
    parser.add_argument(
        '--out_dir',
        type=str,
        help="Directory of output image files",
        required=True
        )
    parser.add_argument(
        '--n_processes',
        type=int,
        required=True
        )
    FLAGS, _ = parser.parse_known_args()
    
    crop_center_blob(FLAGS.in_dir, FLAGS.out_dir, FLAGS.n_processes)
