from keras.models import model_from_json
import os
import utils
import numpy as np
import argparse
from PIL import Image
from scipy import ndimage

# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--img_dir',
    type=str,
    required=True
    )
parser.add_argument(
    '--out_dir',
    type=str,
    required=True
    )
parser.add_argument(
    '--gpu_index',
    type=str,
    required=True
    )
FLAGS, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

# load the models and corresponding weights
with open("../model/vessel/network.json", 'r') as f:
    vessel_model = model_from_json(f.read())
vessel_model.load_weights("../model/vessel/network_weight.h5")

# make directories
if not os.path.isdir(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

# iterate all images
img_size = (640, 640)
filenames = utils.all_files_under(FLAGS.img_dir)
for filename in filenames:
    # load an img (tensor shape of [1,h,w,3])
    img = utils.imagefiles2arrs([filename])
    _, h, w, _ = img.shape
    
    # z score with mean, std (batchsize=1)
    mean = np.mean(img[0, ...][img[0, ..., 0] > 40.0], axis=0)
    std = np.std(img[0, ...][img[0, ..., 0] > 40.0], axis=0)
    img[0, ...] = (img[0, ...] - mean) / std
    assert len(mean) == 3 and len(std) == 3
    
    # run inference & save the result
    vessel = vessel_model.predict(img, batch_size=1)
    Image.fromarray((vessel[0, ..., 0]*255).astype(np.uint8)).save(os.path.join(FLAGS.out_dir, os.path.basename(filename)))