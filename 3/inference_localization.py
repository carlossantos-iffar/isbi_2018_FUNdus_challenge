from keras.models import model_from_json
import os
import utils
import numpy as np
import argparse
import pandas as pd

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

# patch_size for od in original image
patch_size = (1024, 1024)

# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

# load the models and corresponding weights
with open("../model/vessel/network.json", 'r') as f:
    vessel_model = model_from_json(f.read())
vessel_model.load_weights("../model/vessel/network_weight.h5")
with open("../model/od_fovea_localization/network.json", 'r') as f:
    localization_model = model_from_json(f.read())
localization_model.load_weights("../model/od_fovea_localization/network_weight.h5")

# make directories
img_out_dir = os.path.join(FLAGS.out_dir, "pts_check")
if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir)
    
# iterate all images
filepaths = utils.all_files_under(FLAGS.img_dir)
image_no, od, fovea = [], {"x":[], "y":[]}, {"x":[], "y":[]}
for filepath in filepaths:
    assert "IDRiD_" in os.path.basename(filepath)
    # load an img (tensor shape of [1,h,w,3])
    img = utils.imagefiles2arrs([filepath])
    _, h, w, _ = img.shape
    assert h == 2848 and w == 4288
    resized_img = utils.resize_img(img)
    
    # run inference
    vessel = vessel_model.predict(utils.normalize(resized_img, "vessel_segmentation"), batch_size=1)
    predicted_coords, _ = localization_model.predict([utils.normalize(resized_img, "od_from_fundus_vessel"), (vessel * 255).astype(np.uint8) / 255.], batch_size=1)
    
    # convert to coordinates in the original image
    coords = utils.convert_coords_to_original_scale(predicted_coords)
    
    # save the result
    utils.save_imgs_with_pts(img.astype(np.uint8), coords.astype(np.uint16), img_out_dir, os.path.basename(filepath).replace(".jpg", ""))
    od["y"].append(coords[0, 0])
    od["x"].append(coords[0, 1])
    fovea["y"].append(coords[0, 2])
    fovea["x"].append(coords[0, 3])
    image_no.append(os.path.basename(filepath).replace(".jpg", ""))

df_od = pd.DataFrame({'Image No':image_no, 'X-Coordinate':od["x"], 'Y-Coordinate':od["y"]})
df_fovea = pd.DataFrame({'Image No':image_no, 'X-Coordinate':fovea["x"], 'Y-Coordinate':fovea["y"]})
df_od.to_csv(os.path.join(FLAGS.out_dir, "VUNO_OD localization.csv"), index=False)
df_fovea.to_csv(os.path.join(FLAGS.out_dir, "VUNO_fovea localization.csv"), index=False)

