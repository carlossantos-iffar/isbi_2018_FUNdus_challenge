from PIL import Image
from keras.models import model_from_json
import os
import utils
import numpy as np
import argparse
import model

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

tasks = ["EX", "HE", "MA", "SE"]
img_len = {"EX":640, "HE":640, "MA":1280, "SE":640}
# img_dirs = ["../data/original/Apparent_Retinopathy/", "../data/original/No_Apparent_Retinopathy/"]
img_dirs = ["../data/Test/Apparent Retinopathy", "../data/Test/No Apparent Retinopathy"]
# out_home_dir = "../segmentation_results/final_model"
out_home_dir = "../segmentation_results/final_model_test"
normalized_methods = {"EX":"rescale", "HE":"rescale_mean_subtract", "MA":"rescale_mean_subtract", "SE":"rescale"}
for model_dir in ["model"]:
    for task in tasks:
        # load the models and corresponding weights
        load_model_dir = "../{}/{}".format(model_dir, task)
        network_file = utils.all_files_under(load_model_dir, extension=".json")
        weight_file = utils.all_files_under(load_model_dir, extension=".h5")
        if task == "MA":
            segmentation_model = model.unet_atrous_sr(3, 3, 2, (1280, 1280), 32) 
        else:
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
                resized_img = utils.resize_img(img, img_len[task])
               
                # run inference
                segmented = segmentation_model.predict(utils.normalize(resized_img, normalized_methods[task]), batch_size=1)

                # size up
                segmented = segmented[0, ..., 0]
                segmented_sizeup = utils.sizeup(segmented, img_len[task])
                
                # save the result
                Image.fromarray((segmented_sizeup * 255).astype(np.uint8)).save(os.path.join(out_dir, os.path.basename(filename)))
        
