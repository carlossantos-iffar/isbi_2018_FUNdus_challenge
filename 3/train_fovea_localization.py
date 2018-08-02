import numpy as np
from model import localize_fv
import utils
import os
import argparse
from keras import backend as K
import iterator_localization
import sys

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
FLAGS, _ = parser.parse_known_args()

# training settings 
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
n_epochs = 1000
validation_epochs = range(0, 500, 50) + range(500, 800, 10) + range(800, 1000, 1)
batch_size = FLAGS.batch_size
schedules = {'lr':{'0': 0.001, '500':0.0001}}

# set misc paths
img_size = (640, 640)
feature_map_size = (20, 20)
model_out_dir = "models_fovea"
vessel_train_dir = "../outputs/vessels_localization/training"
vessel_val_dir = "../outputs/vessels_localization/validation"
fundus_train_dir = "../data/localization/training"
fundus_val_dir = "../data/localization/validation"
train_img_check_dir = "../input_checks/localization/train/imgs"
val_img_check_dir = "../input_checks/localization/validation/imgs"
img_out_dir = "localized_fovea"
od_label_path = "../data/IDRiD_OD_Center_Training_set.csv"
fovea_label_path = "../data/IDRiD_Fovea_Center_Training_set.csv"
if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir)
if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir)
 
# set iterators for training and validation
train_batch_fetcher = iterator_localization.TrainBatchFetcher(fundus_train_dir, vessel_train_dir, od_label_path, fovea_label_path, FLAGS.batch_size)
validation_batch_fetcher = iterator_localization.ValidationBatchFetcher(fundus_val_dir, vessel_val_dir, od_label_path, fovea_label_path, FLAGS.batch_size)

# create networks
network = localize_fv(img_size) 
network.summary()
with open(os.path.join(model_out_dir, "network.json"), 'w') as f:
    f.write(network.to_json())

# start training
scheduler = utils.Scheduler(schedules)
check_train_batch, check_validation_batch = True, True
for epoch in range(n_epochs):
    # update step sizes, learning rates
    scheduler.update_steps(epoch)
    K.set_value(network.optimizer.lr, scheduler.get_lr())    
    
    # train on the training set
    losses_dist, losses_dist_vessel = [], []
    for fundus_batch, vessel_batch, coords_batch, filenames in train_batch_fetcher():
        # fundus_batch : (batch_size, 640, 640, 3)
        # vessel_batch : (batch_size, 640, 640, 1)
        # coords_batch : (batch_size, 4) (od_y,od_x,fv_y,fv_x)
        total_loss, loss_dist, loss_dist_vessel = network.train_on_batch([fundus_batch, vessel_batch], [coords_batch[:, 2:], coords_batch[:, 2:]])
        losses_dist += [loss_dist] * len(filenames)
        losses_dist_vessel += [loss_dist_vessel] * len(filenames)
    print "loss_dist: {}, loss_dist_vessel: {},".format(np.mean(losses_dist), np.mean(losses_dist_vessel))
    
    # evaluate on validation set
    if epoch in validation_epochs:
        fundus_list, coords_list, predicted_coords_list, predicted_coords_vessel_list = [], [], [], []
        for fundus_batch, vessel_batch, coords_batch, filenames in validation_batch_fetcher():
            predicted_coords, predicted_coords_vessel = network.predict([fundus_batch, vessel_batch], batch_size=batch_size, verbose=0)
            coords_list.append(coords_batch[:,2:])
            predicted_coords_list.append(predicted_coords)
            predicted_coords_vessel_list.append(predicted_coords_vessel)
            fundus_list.append(fundus_batch)
        val_predicted_coords = np.concatenate(predicted_coords_list, axis=0)
        val_coords = np.concatenate(coords_list, axis=0)
        val_fundus = np.concatenate(fundus_list, axis=0)
        Euclidean_fovea = utils.mean_Euclidean_fovea(val_coords, val_predicted_coords)
        utils.print_metrics(epoch + 1, Euclidean_fovea=Euclidean_fovea)
            
        # save the weight
        network.save_weights(os.path.join(model_out_dir, "network_{}.h5".format(epoch + 1)))
        
        # save validation results
        utils.save_imgs_with_pts_fovea(val_fundus, val_predicted_coords, img_out_dir, epoch)

    sys.stdout.flush()
