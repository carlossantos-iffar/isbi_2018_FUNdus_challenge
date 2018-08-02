import numpy as np
from model import segment_fv
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
n_epochs = 500
validation_epochs = range(0, 300, 10) + range(300, 500, 1)
batch_size = FLAGS.batch_size
schedules = {'lr':{'0': 0.001, '300':0.0001}}

# set misc paths
img_size = (640, 640)
feature_map_size = (20, 20)
model_out_dir = "models_fovea_segmentation"
vessel_train_dir = "../outputs/vessels_localization/training"
vessel_val_dir = "../outputs/vessels_localization/validation"
fundus_train_dir = "../data/localization/training"
fundus_val_dir = "../data/localization/validation"
train_img_check_dir = "../input_checks/fovea_segmentation/train/imgs"
val_img_check_dir = "../input_checks/fovea_segmentation/validation/imgs"
img_out_dir = "localized_fovea_segmentation"
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
network = segment_fv(img_size) 
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
    losses_seg, losses_seg_vessel = [], []
    for fundus_batch, vessel_batch, coords_batch, filenames in train_batch_fetcher():
        # fundus_batch : (batch_size, 640, 640, 3)
        # vessel_batch : (batch_size, 640, 640, 1)
        # coords_batch : (batch_size, 4) (od_y,od_x,fv_y,fv_x)
        fovea_mask, fovea_mask_vessel=utils.fovea_mask(coords_batch[:,2:])
        if check_train_batch:
            utils.save_fig_fovea_segmentation(fundus_batch, fovea_mask, train_img_check_dir)
            check_train_batch=False
        total_loss, loss_seg, loss_seg_vessel = network.train_on_batch([fundus_batch, vessel_batch], [fovea_mask, fovea_mask_vessel])
        losses_seg += [loss_seg] * len(filenames)
        losses_seg_vessel += [loss_seg_vessel] * len(filenames)
    print "training losses_seg: {}, training losses_seg_vessel: {},".format(np.mean(losses_seg), np.mean(losses_seg_vessel))
    
    # evaluate on validation set
    if epoch in validation_epochs:
        losses_seg, losses_seg_vessel, predicted_masks, fundus_list = [], [], [], []
        for fundus_batch, vessel_batch, coords_batch, filenames in validation_batch_fetcher():
            fovea_mask, fovea_mask_vessel=utils.fovea_mask(coords_batch[:,2:])
            if check_validation_batch:
                utils.save_fig_fovea_segmentation(fundus_batch, fovea_mask, val_img_check_dir)
                check_validation_batch=False
            total_loss, loss_seg, loss_seg_vessel=network.evaluate([fundus_batch, vessel_batch], [fovea_mask, fovea_mask_vessel], batch_size=batch_size, verbose=0)
            losses_seg += [loss_seg] * len(filenames)
            losses_seg_vessel += [loss_seg_vessel] * len(filenames)
            predicted_mask, predicted_mask_vessel = network.predict([fundus_batch, vessel_batch], batch_size=batch_size, verbose=0)
            predicted_masks.append(predicted_mask)
            fundus_list.append(fundus_batch)
        val_fundus = np.concatenate(fundus_list, axis=0)
        val_predicted_masks = np.concatenate(predicted_masks, axis=0)
        print "validation losses_seg: {}, validation losses_seg_vessel: {},".format(np.mean(losses_seg), np.mean(losses_seg_vessel))

        # save the weight
        network.save_weights(os.path.join(model_out_dir, "network_{}.h5".format(epoch + 1)))
        
        # save validation results
        utils.save_fig_fovea_segmentation(val_fundus, val_predicted_masks, img_out_dir)

    sys.stdout.flush()
