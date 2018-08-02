import numpy as np
import model
import utils
import os
import argparse
from keras import backend as K
import iterator_dr_from_features
import sys
from keras.models import model_from_json
import time

# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpu_index',
    type=str,
    help="gpu index",
    required=True
    )
parser.add_argument(
    '--batch_size',
    type=int,
    help="batch size",
    required=True
    )
parser.add_argument(
    '--loss_type',
    type=str,
    required=True
    )
parser.add_argument(
    '--data_ratio',
    type=float,
    required=False
    )
parser.add_argument(
    '--load_model_dir',
    type=str,
    required=False
    )
FLAGS, _ = parser.parse_known_args()

# training settings 
val_ratio = 0.1
n_epochs = 100
schedules = {'lr':{'0':0.0001}}
validation_epochs = range(0, 100, 1)
batch_size = FLAGS.batch_size
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
img_size = (640, 640)

# set misc paths
# fundus_dirs = ["../data/kaggle_DR_train/preprocessed", "../data/kaggle_DR_test/preprocessed", "../data/Training_Set_preprocessed"]
fundus_dirs = ["../data/kaggle_DR_test/preprocessed_640"]
grade_path = "../data/all_labels_tmp.csv"
model_out_dir = "/nfs/jaemin/isbi/2/models/dr/only_features_{}_{}".format(FLAGS.data_ratio, FLAGS.loss_type)
features_home = "../data/bottleneck_features/"
EX_segmentor_dir = "../model/EX_segmentor"
HE_segmentor_dir = "../model/HE_segmentor"
MA_segmentor_dir = "../model/MA_segmentor"
SE_segmentor_dir = "../model/SE_segmentor"

if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir)

# set iterators for training and validation
training_set, validation_set = utils.split_dr(fundus_dirs, grade_path, val_ratio)
if FLAGS.data_ratio:
    n_train = int(len(training_set[0]) * FLAGS.data_ratio)
    n_val = int(len(validation_set[0]) * FLAGS.data_ratio)
    training_set = (training_set[0][:n_train], training_set[1][:n_train])
    validation_set = (validation_set[0][:n_val], validation_set[1][:n_val])
class_weight = utils.class_weight(training_set[-1])
# class_weight = [1, 2, 2, 4, 4]
train_batch_fetcher = iterator_dr_from_features.TrainBatchFetcher(training_set, batch_size, features_home, class_weight)
val_batch_fetcher = iterator_dr_from_features.ValidationBatchFetcher(validation_set, batch_size, features_home)

# create networks
if FLAGS.load_model_dir:
    network_file = utils.all_files_under(FLAGS.load_model_dir, extension=".json")
    weight_file = utils.all_files_under(FLAGS.load_model_dir, extension=".h5")
    assert len(network_file) == 1 and len(weight_file) == 1
    with open(network_file[0], 'r') as f:
        network = model_from_json(f.read())
    network.load_weights(weight_file[0])
    network = model.set_optimizer(network)
else:
    network = model.dr_classifier_from_features_no_conv_branch(FLAGS.loss_type)
network.summary()
with open(os.path.join(model_out_dir, "network.json"), 'w') as f:
    f.write(network.to_json())

# start training
scheduler = utils.Scheduler(schedules)
for epoch in range(n_epochs):
    # update step sizes, learning rates
    scheduler.update_steps(epoch)
    K.set_value(network.optimizer.lr, scheduler.get_lr())
    
    # train on the training set
    start_time = time.time()
    losses, accs = [], []
    n_itr = 0
    for fnames, ex_features, he_features, ma_features, se_features, fundus_rescale_mean_subtract, grades in train_batch_fetcher():
        loss, acc = network.train_on_batch([ex_features, he_features,
                                            ma_features, se_features], grades)
        losses += [loss] * fundus_rescale_mean_subtract.shape[0]
        accs += [acc] * fundus_rescale_mean_subtract.shape[0]
        n_itr += 1
        if n_itr % 100 == 0:
            print loss
    utils.print_metrics(epoch + 1, training_loss=np.mean(losses), training_acc=np.mean(accs))
  
    # evaluate on the validation set
    if epoch in validation_epochs:
        pred_grades, true_grades = [], []
        losses, accs = [], []
        for fnames, ex_features, he_features, ma_features, se_features, fundus_rescale_mean_subtract, grades in val_batch_fetcher():
            pred = network.predict([ex_features, he_features,
                                            ma_features, se_features], batch_size=batch_size, verbose=0)
            loss, acc = network.evaluate([ex_features, he_features,
                                            ma_features, se_features], grades, batch_size=batch_size, verbose=0)
            pred_grades += pred.tolist()
            true_grades += grades.tolist()
            losses += [loss] * ex_features.shape[0]
            accs += [acc] * ex_features.shape[0]
        
        utils.print_confusion_matrix(true_grades, pred_grades, "DR")
        utils.print_metrics(epoch + 1, acc=np.mean(accs), validation_losses=np.mean(losses))

        # save the weight
        if epoch in validation_epochs:
            network.save_weights(os.path.join(model_out_dir, "network_{}.h5".format(epoch + 1)))
    
    duration = time.time() - start_time
    print "duration for {}th epoch: {}s".format(epoch + 1, duration)    
    sys.stdout.flush()
