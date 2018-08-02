import numpy as np
from model import od_from_fundus_vessel_v4
import utils
import os
from PIL import Image
import argparse
from keras import backend as K
import iterator_segmentation
import sys
from scipy.spatial.distance import dice

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
n_epochs = 800
batch_size = FLAGS.batch_size
schedules = {'lr':{'0':2e-4, '400':2e-5}}

# set misc paths
img_size = (640, 640)
img_out_dir = "segmentation_results"
model_out_dir = "models"
vessel_train_dir = "../outputs/vessels_segmentation/training"
vessel_val_dir = "../outputs/vessels_segmentation/validation"
fundus_train_dir = "../data/disc_segmentation/training/images"
fundus_val_dir = "../data/disc_segmentation/validation/images"
mask_train_dir = "../data/disc_segmentation/training/masks"
mask_val_dir = "../data/disc_segmentation/validation/masks"
train_img_check_dir = "../input_checks/segmentation/train/"
val_img_check_dir = "../input_checks/segmentation/validation/"
if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir)
if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir)

# set iterators for training and validation
train_fundus_fns, train_vessel_fns, train_mask_fns = utils.all_files_under(fundus_train_dir), utils.all_files_under(vessel_train_dir), utils.all_files_under(mask_train_dir)
val_fundus_fns, val_vessel_fns, val_mask_fns = utils.all_files_under(fundus_val_dir), utils.all_files_under(vessel_val_dir), utils.all_files_under(mask_val_dir)
train_batch_fetcher = iterator_segmentation.TrainBatchFetcher(train_fundus_fns, train_vessel_fns, train_mask_fns, FLAGS.batch_size)
val_imgs, val_vessels, val_masks = utils.set_validationset_fundus_vessel(val_fundus_fns, val_vessel_fns, val_mask_fns)

# create networks
# network = od_from_fundus_vessel(img_size) 
# network = od_from_fundus_vessel_v2(img_size) 
# network = od_from_fundus_vessel_v3(img_size) 
network = od_from_fundus_vessel_v4(img_size) 
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
    losses_all, losses_vessel = [], []
    for filenames, imgs, vessels, segs in train_batch_fetcher():
        if check_train_batch:
            utils.check_input(imgs, segs, train_img_check_dir)
#             utils.check_input(vessels, segs, train_img_check_dir)
            check_train_batch = False
        total_loss, loss_all, loss_vessel = network.train_on_batch([imgs, vessels], [segs, segs[:, ::32, ::32, :]])
        losses_all += [loss_all] * len(filenames)
        losses_vessel += [loss_vessel] * len(filenames)
    print "loss_all: {}, loss_vessel: {}".format(np.mean(losses_all), np.mean(losses_vessel))
    
    # evaluate on validation set
    if check_validation_batch:
        utils.check_input(val_imgs, val_masks, val_img_check_dir)
#         utils.check_input(val_vessels, val_masks, val_img_check_dir)
        check_test_batch = False
    val_generated_masks_f_v, val_generate_masks_v = network.predict([val_imgs, val_vessels], batch_size=batch_size, verbose=0)
    auroc = utils.AUC_ROC(val_masks, val_generated_masks_f_v)
    aupr = utils.AUC_PR(val_masks, val_generated_masks_f_v)
    val_generated_masks_f_v = np.round(val_generated_masks_f_v)
    cm, spe, sen, dice_val, jaccard_val = utils.seg_metrics(val_masks, val_generated_masks_f_v)
    utils.print_metrics(epoch + 1, jaccard_val=jaccard_val, dice_val=dice_val, sen=sen, auroc=auroc, aupr=aupr)
    
    # save the weight
    network.save_weights(os.path.join(model_out_dir, "network_{}.h5".format(epoch)))
    
    # save validation results
    for index in range(val_generated_masks_f_v.shape[0]):
        Image.fromarray((val_generated_masks_f_v[index, ..., 0] * 255).astype(np.uint8)).save(os.path.join(img_out_dir, str(epoch) + "_{:02}_od_from_fundus_vessel.png".format(index + 1)))
        Image.fromarray((val_generate_masks_v[index, ..., 0] * 255).astype(np.uint8)).save(os.path.join(img_out_dir, str(epoch) + "_{:02}_od_from_vessel.png".format(index + 1)))
sys.stdout.flush()
