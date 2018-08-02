import numpy as np
import utils
import os
import argparse
import iterator_dr
import pandas as pd
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix

# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpu_index',
    type=str,
    help="gpu index",
    required=True
    )
FLAGS, _ = parser.parse_known_args()

# training settings 
val_ratio = 1.0
batch_size = 1
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

# set misc paths
fundus_dirs = ["../data/Training_Set_preprocessed"]
grade_path = "../data/all_labels.csv"
load_model_dir = "../model/DR_jaemin_net"

# set iterators for training and validation
training_set, validation_set = utils.split_dr(fundus_dirs, grade_path, val_ratio)
val_batch_fetcher = iterator_dr.ValidationBatchFetcher(validation_set, batch_size)

# load network
network_file = utils.all_files_under(load_model_dir, extension=".json")
weight_file = utils.all_files_under(load_model_dir, extension=".h5")
assert len(network_file) == 1 and len(weight_file) == 1
with open(network_file[0], 'r') as f:
    network = model_from_json(f.read())
network.load_weights(weight_file[0])

# run inference
filepaths, filenames, pred_grades, true_grades = [], [], [], []
for fnames, fundus_rescale, fundus_rescale_mean_subtract, grades in val_batch_fetcher():
    pred = network.predict(fundus_rescale_mean_subtract, batch_size=batch_size, verbose=0)
    pred_grades += pred[0].tolist()
    true_grades += grades.tolist()
    filenames += [os.path.basename(fname).replace(".tif", "") for fname in fnames.tolist()]
    filepaths += fnames.tolist()

final_prediction=utils.adjust_threshold(true_grades, pred_grades)
df = pd.DataFrame({"Image No":filenames, "DR Grade":final_prediction})
df.to_csv("VRT_Disease_Grading_DR.csv", index=False)
# segmented_dir_tempalte = "../outputs//{}/"
# ori_img_dir = "../data/merged_training_set/"
# utils.save_wrong_files(true_grades, pred_grades, filepaths, segmented_dir_tempalte, ori_img_dir)

utils.print_confusion_matrix(true_grades, final_prediction, "DR")