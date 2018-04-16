import numpy as np
import utils
import os
import argparse
import iterator_dr_640
from PIL import Image

# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpu_index',
    type=str,
    help="gpu index",
    required=True
    )
parser.add_argument(
    '--load_model_dir',
    type=str,
    required=False
    )
FLAGS, _ = parser.parse_known_args()

# training settings 
val_ratio = 1.0
validation_epochs = range(0, 100, 1)
batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

# set misc paths
# fundus_dirs = ["../data/Training_Set_preprocessed", "../data/kaggle_DR_test/preprocessed_640/"]
fundus_dirs = ["../data/kaggle_DR_test/preprocessed_640/"]
# grade_path = "../data/all_labels.csv"
grade_path = "../data/all_labels_tmp.csv"
img_out_dir = "../outputs/lesion_segmentation"
EX_segmentor_dir = "../model/EX_segmentor"
HE_segmentor_dir = "../model/HE_segmentor"
MA_segmentor_dir = "../model/MA_segmentor"
SE_segmentor_dir = "../model/SE_segmentor"

for task in ["EX", "HE", "MA", "SE"]:
    if not os.path.isdir(os.path.join(img_out_dir, task)):
        os.makedirs(os.path.join(img_out_dir, task))

# set iterators for training and validation
training_set, validation_set = utils.split_dr(fundus_dirs, grade_path, val_ratio)
val_batch_fetcher = iterator_dr_640.ValidationBatchFetcher(validation_set, batch_size)

EX_segmentor = utils.load_network(EX_segmentor_dir)
HE_segmentor = utils.load_network(HE_segmentor_dir)
MA_segmentor = utils.load_network(MA_segmentor_dir)
SE_segmentor = utils.load_network(SE_segmentor_dir)

# normalization_method
# EX, SE: rescale
# HE, MA: rescale_mean_subtract
# network input order: ex, he, ma, se 
for fnames, fundus_rescale, fundus_rescale_mean_subtract, grades in val_batch_fetcher():
    # debugging purpose
    ex_arr = EX_segmentor.predict(fundus_rescale, batch_size=batch_size, verbose=0)
    he_arr = HE_segmentor.predict(fundus_rescale_mean_subtract, batch_size=batch_size, verbose=0)
    ma_arr = MA_segmentor.predict(fundus_rescale_mean_subtract, batch_size=batch_size, verbose=0)
    se_arr = SE_segmentor.predict(fundus_rescale, batch_size=batch_size, verbose=0)
    
    for index in range(ex_arr.shape[0]):
        ex = ex_arr[index, ..., 0]
        he = he_arr[index, ..., 0]
        ma = ma_arr[index, ..., 0]
        se = se_arr[index, ..., 0]
        
        Image.fromarray((ex*255).astype(np.uint8)).save(os.path.join(img_out_dir, "EX", os.path.basename(fnames[index])))
        Image.fromarray((he*255).astype(np.uint8)).save(os.path.join(img_out_dir, "HE", os.path.basename(fnames[index])))
        Image.fromarray((ma*255).astype(np.uint8)).save(os.path.join(img_out_dir, "MA", os.path.basename(fnames[index])))
        Image.fromarray((se*255).astype(np.uint8)).save(os.path.join(img_out_dir, "SE", os.path.basename(fnames[index])))
