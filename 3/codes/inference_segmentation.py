from PIL import Image
from keras.models import model_from_json
import os
import utils
import numpy as np
import argparse

# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpu_index',
    type=str,
    required=True
    )
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
FLAGS, _ = parser.parse_known_args()

# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

# load the models and corresponding weights
with open("../model/vessel/network.json", 'r') as f:
    vessel_model = model_from_json(f.read())
vessel_model.load_weights("../model/vessel/network_weight.h5")
with open("../model/od_from_fundus_vessel/network.json", 'r') as f:
    od_from_fundus_vessel_model = model_from_json(f.read())
od_from_fundus_vessel_model.load_weights("../model/od_from_fundus_vessel/network_weight.h5")

# make directories
out_vessel_dir_visualzation = os.path.join(FLAGS.out_dir, "vessel_visualization")
out_od_dir_visualization = os.path.join(FLAGS.out_dir, "od_from_fundus_vessel_visualization")
out_od_dir = os.path.join(FLAGS.out_dir, "od_from_fundus_vessel")
if not os.path.isdir(out_vessel_dir_visualzation):
    os.makedirs(out_vessel_dir_visualzation)
if not os.path.isdir(out_od_dir):
    os.makedirs(out_od_dir)
if not os.path.isdir(out_od_dir_visualization):
    os.makedirs(out_od_dir_visualization)
    
# iterate all images
filenames = utils.all_files_under(FLAGS.img_dir)
for filename in filenames:
    assert "IDRiD_" in os.path.basename(filename)
    # load an img (tensor shape of [1,h,w,3])
    img = utils.imagefiles2arrs([filename])
    _, h, w, _ = img.shape
    assert h == 2848 and w == 4288
    resized_img = utils.resize_img(img)
    
    # run inference
    vessel = vessel_model.predict(utils.normalize(resized_img, "vessel_segmentation"), batch_size=1)
    od_seg, _ = od_from_fundus_vessel_model.predict([utils.normalize(resized_img, "od_from_fundus_vessel"), (vessel * 255).astype(np.uint8) / 255.], batch_size=1)
    
    # cut by threshold
    od_seg = od_seg[0, ..., 0]
    od_seg_zoomed_to_ori_scale = utils.sizeup(od_seg)
#     od_seg_zoomed_to_ori_scale = utils.cut_by_threshold(od_seg_zoomed_to_ori_scale, threshold=0.5)
    
    # save the result
    Image.fromarray((vessel[0, ..., 0] * 255).astype(np.uint8)).save(os.path.join(out_vessel_dir_visualzation, os.path.basename(filename)))
#     Image.fromarray((od_seg_zoomed_to_ori_scale).astype(np.uint8)).save(os.path.join(out_od_dir, os.path.basename(filename)))
    Image.fromarray((od_seg_zoomed_to_ori_scale * 255).astype(np.uint8)).save(os.path.join(out_od_dir_visualization, os.path.basename(filename)))

