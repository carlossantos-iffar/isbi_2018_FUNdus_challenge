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
FLAGS, _ = parser.parse_known_args()

# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

tasks = ["EX"]
img_dirs = ["../data/original/Apparent_Retinopathy/", "../data/original/No_Apparent_Retinopathy/"]
out_home_dir = "../segmentation_results_for_sc2/"
for model_dir in ["model_for_sc2"]:
    for task in tasks:
        # load the models and corresponding weights
        load_model_dir = "../{}/{}".format(model_dir, task)
        network_file = utils.all_files_under(load_model_dir, extension=".json")
        weight_file = utils.all_files_under(load_model_dir, extension=".h5")
        assert len(network_file) == 1 and len(weight_file) == 1
        with open(network_file[0], 'r') as f:
            segmentation_model = model_from_json(f.read())
        segmentation_model.load_weights(weight_file[0])
        
        # make directories
        out_dir = os.path.join(out_home_dir, task)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        # iterate all images
        for img_dir in img_dirs:
            filenames = utils.all_files_under(img_dir)
            for filename in filenames:
                assert "IDRiD_" in os.path.basename(filename)
                # load an img (tensor shape of [1,h,w,3])
                img = utils.imagefiles2arrs([filename])
                _, h, w, _ = img.shape
                assert h == 2848 and w == 4288
                resized_img = utils.resize_img(img)
            
                # run inference
                segmented = segmentation_model.predict(utils.normalize(resized_img), batch_size=1)
                
                # cut by threshold
                segmented = segmented[0, ..., 0]
                segmented_sizeup = utils.sizeup(segmented)
                
                # save the result
                Image.fromarray((segmented_sizeup * 255).astype(np.uint8)).save(os.path.join(out_dir, os.path.basename(filename)))
        
