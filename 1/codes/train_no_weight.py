import numpy as np
import model
import utils
import os
from PIL import Image
import argparse
from keras import backend as K
import iterator
import sys
from keras.models import model_from_json

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
    '--task',
    type=str,
    required=True
    )
parser.add_argument(
    '--load_model_dir',
    type=str,
    required=False,
    default=None
    )
FLAGS, _ = parser.parse_known_args()

# training settings 
depths = {"EX":7, "HE":7, "MA":3, "SE":5}
depth = depths[FLAGS.task]
atrous_depth = depth
validation_epochs = range(0, 500) if FLAGS.load_model_dir else range(0, 400, 1)
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
n_epochs = 300
batch_size = FLAGS.batch_size
schedules = {'lr':{'0':2e-4, '250':2e-5}}

# set misc paths
train_patch_size = (640, 640)  # cropped img size:  (2848, 3500)
validation_patch_size = (640, 640)
img_out_template = "../segmentation_results/unet/segmentation_results_{}/{}"
model_out_template = "../models/unet/model_{}/{}"
img_out_dir = img_out_template.format(atrous_depth, FLAGS.task + "_no_weight")
model_out_dir = model_out_template.format(atrous_depth, FLAGS.task + "_no_weight")
train_img_check_dir = "../outputs/input_checks/train"
val_img_check_dir = "../outputs/input_checks/validation"
if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir)
if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir)
if not os.path.isdir(train_img_check_dir):
    os.makedirs(train_img_check_dir)
if not os.path.isdir(val_img_check_dir):
    os.makedirs(val_img_check_dir)
home_dir = "../data/preprocessed/"
AR = os.path.join(home_dir, "Apparent_Retinopathy")
NAR = os.path.join(home_dir, "No_Apparent_Retinopathy")
seg_paths = {"EX":os.path.join(home_dir, "EX"), "HE":os.path.join(home_dir, "HE"),
           "MA":os.path.join(home_dir, "MA"), "SE":os.path.join(home_dir, "SE")}

# set iterators for training and validation
train_pairs, val_pairs = utils.split_dataset(AR, NAR, seg_paths[FLAGS.task], 0.1, exclude_NAR=False)
train_batch_fetcher = iterator.BatchFetcher(train_pairs, FLAGS.batch_size, train_patch_size, augment=True)
val_batch_fetcher = iterator.BatchFetcher(val_pairs, FLAGS.batch_size, validation_patch_size, augment=False)

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
    network = model.unet_atrous(depth, atrous_depth) 
network.summary()
with open(os.path.join(model_out_dir, "network.json"), 'w') as f:
    f.write(network.to_json())

# start training
scheduler = utils.Scheduler(schedules)
check_train_batch, check_validation_batch = True, True
best_aupr = 0
for epoch in range(n_epochs):
    # update step sizes, learning rates
    scheduler.update_steps(epoch)
    K.set_value(network.optimizer.lr, scheduler.get_lr())
    
    # train on the training set
    losses = []
    for imgs, segs in train_batch_fetcher():
        if check_train_batch:
            utils.check_input(imgs, segs, train_img_check_dir)
            check_train_batch = False
        loss = network.train_on_batch(imgs, segs)
        losses += [loss] * imgs.shape[0]
    utils.print_metrics(epoch + 1, training_loss=np.mean(losses))
    
    # evaluate on the validation set
    if epoch in validation_epochs:
        losses, fundus_imgs, gt_masks, pred_masks = [], [], [], []
        for imgs, segs in val_batch_fetcher():
            if check_validation_batch:
                utils.check_input(imgs, segs, train_img_check_dir)
                check_validation_batch = False
            pred = network.predict(imgs, batch_size=batch_size, verbose=0)
            loss = network.evaluate(imgs, segs, batch_size=batch_size, verbose=0)
            losses += [loss] * imgs.shape[0]
            pred_masks.append(pred)
            gt_masks.append(segs)
            fundus_imgs.append(imgs)
        
        pred_masks = np.concatenate(pred_masks, axis=0)
        gt_masks = np.concatenate(gt_masks, axis=0)
        fundus_imgs = np.concatenate(fundus_imgs, axis=0)
        
        # evaluate results
        auroc = utils.AUC_ROC(gt_masks, pred_masks)
        aupr = utils.AUC_PR(gt_masks, pred_masks)
        utils.print_metrics(epoch + 1, auroc=auroc, aupr=aupr, validation_losses=np.mean(losses))
    
        # save the weight
        if aupr > best_aupr:
            network.save_weights(os.path.join(model_out_dir, "network_{}.h5".format(epoch + 1)))
            best_aupr = aupr
        
        # save validation results
        for index in range(pred_masks.shape[0]):
            Image.fromarray((pred_masks[index, ..., 0] * 255).astype(np.uint8)).save(os.path.join(img_out_dir, str(epoch + 1) + "_{:02}_segmented.png".format(index + 1)))
            Image.fromarray((gt_masks[index, ..., 0] * 255).astype(np.uint8)).save(os.path.join(img_out_dir, str(epoch + 1) + "_{:02}_gt.png".format(index + 1)))
            Image.fromarray((fundus_imgs[index, ...] * 255).astype(np.uint8)).save(os.path.join(img_out_dir, str(epoch + 1) + "_{:02}_fundus_patch.png".format(index + 1)))
    sys.stdout.flush()
