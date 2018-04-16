import utils
import os
import numpy as np
from sklearn.metrics.classification import confusion_matrix
import argparse
from skimage import measure
from scipy.spatial.distance import dice
from keras.models import model_from_json


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

gt_dir = "../data/original/OD_Segmentation_Training_Set/"
gt_filenames = utils.all_files_under(gt_dir)
gt = utils.imagefiles2arrs(gt_filenames).astype(np.uint8)

# load the models and corresponding weights
with open("../model/vessel/network.json", 'r') as f:
    vessel_model = model_from_json(f.read())
vessel_model.load_weights("../model/vessel/network_weight.h5")
with open("../model/od_from_fundus_vessel/network.json", 'r') as f:
    od_from_fundus_vessel_model = model_from_json(f.read())
od_from_fundus_vessel_model.load_weights("../model/od_from_fundus_vessel/network_weight.h5")

# iterate all images
filenames = utils.all_files_under("../data/original/Training_c/")
pred = np.zeros(gt.shape)
for index, filename in enumerate(filenames):
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
    pred[index, ...] = od_seg_zoomed_to_ori_scale

print utils.segmentation_optimal_threshold(gt, pred)
