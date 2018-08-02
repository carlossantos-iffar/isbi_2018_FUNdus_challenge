import numpy as np
import model
import utils
import os
import argparse
from keras import backend as K
import iterator_dme
import sys
from keras.models import model_from_json
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
    '--batch_size',
    type=int,
    help="batch size",
    required=True
    )
parser.add_argument(
    '--grade_type',
    type=str,
    required=True
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
schedules = {'lr':{'0':0.001, '500':0.0001}}
validation_epochs = range(0, 100, 1)
batch_size = FLAGS.batch_size
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
img_size = (640, 640)

# set misc paths
fundus_dir = "../data/merged_training_set"
vessel_dir = "../data/merged_vessel"
grade_path = "../data/merged_labels.csv"
img_out_dir = "/nfs/jaemin/isbi/2/img_outputs"
model_out_dir = "/nfs/jaemin/isbi/2/models/{}".format(FLAGS.grade_type)
train_img_check_dir = "../outputs/input_checks/train"
val_img_check_dir = "../outputs/input_checks/validation"
EX_segmentor_dir = "../model/EX_segmentor"
fovea_localizer_dir = "../model/fovea"

if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir)
if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir)
if not os.path.isdir(train_img_check_dir):
    os.makedirs(train_img_check_dir)
if not os.path.isdir(val_img_check_dir):
    os.makedirs(val_img_check_dir)

# set iterators for training and validation
training_set, validation_set = utils.split(fundus_dir, vessel_dir, grade_path, FLAGS.grade_type, val_ratio)
# class_weight = utils.class_weight(training_set[-1])
class_weight = (1. / 8, 1. / 2, 1. / 4)
train_batch_fetcher = iterator_dme.TrainBatchFetcher(training_set, batch_size, FLAGS.grade_type, class_weight)
val_batch_fetcher = iterator_dme.ValidationBatchFetcher(validation_set, batch_size, FLAGS.grade_type)

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
    EX_segmentor = utils.load_network(EX_segmentor_dir)
    fovea_localizer = utils.load_network(fovea_localizer_dir)
    network = model.dme_classifier(EX_segmentor, fovea_localizer)
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
    losses, accs = [], []
    for fnames, imgs_mean_subt, imgs_z, vessels, grades_onehot in train_batch_fetcher():
        if check_train_batch:
            utils.check_input(imgs_mean_subt, imgs_z, vessels, train_img_check_dir)
            check_train_batch = False
        loss, acc = network.train_on_batch([imgs_mean_subt, imgs_z, vessels], grades_onehot)
        losses += [loss] * imgs_mean_subt.shape[0]
        accs += [acc] * imgs_mean_subt.shape[0]
    utils.print_metrics(epoch + 1, training_loss=np.mean(losses), training_acc=np.mean(accs))
  
    # debugging purpose
#     print fnames
#     segmented = EX_segmentor.predict(imgs_mean_subt, batch_size=batch_size, verbose=0)
#     for index in range(segmented.shape[0]):
#         Image.fromarray((segmented[index, ..., 0] * 255).astype(np.uint8)).save(os.path.join(img_out_dir, str(epoch + 1) + "_{:02}_segmented.png".format(index + 1)))
#     fovea, fovea_from_vessel = fovea_localizer.predict([imgs_z, vessels], batch_size=batch_size, verbose=0)
#     utils.save_imgs_with_pts_fovea(imgs_z, fovea, img_out_dir, epoch)
#     exit(1)
    
    # evaluate on the validation set
    if epoch in validation_epochs:
        losses, accs = [], []
        pred_grades, true_grades = [], []
        for fnames, imgs_mean_subt, imgs_z, vessels, grades_onehot  in val_batch_fetcher():
            if check_validation_batch:
                utils.check_input(imgs_mean_subt, imgs_z, vessels, val_img_check_dir)
                check_validation_batch = False
            pred = network.predict([imgs_mean_subt, imgs_z, vessels], batch_size=batch_size, verbose=0)
            pred_grades += np.argmax(pred, axis=1).tolist()
            true_grades += np.argmax(grades_onehot, axis=1).tolist()
            loss, acc = network.evaluate([imgs_mean_subt, imgs_z, vessels], grades_onehot, batch_size=batch_size, verbose=0)
            losses += [loss] * imgs_mean_subt.shape[0]
            accs += [acc] * imgs_mean_subt.shape[0]
        utils.print_metrics(epoch + 1, validation_loss=np.mean(losses), validation_acc=np.mean(accs))
        utils.print_confusion_matrix(true_grades, pred_grades, FLAGS.grade_type)
    
        # save the weight
        if epoch in validation_epochs:
            network.save_weights(os.path.join(model_out_dir, "network_{}.h5".format(epoch + 1)))
        
    sys.stdout.flush()
