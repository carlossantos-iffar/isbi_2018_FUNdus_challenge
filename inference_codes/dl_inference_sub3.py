import os
import utils_sub3
import numpy as np
import pandas as pd
import time
from PIL import Image


def localization_inference(img_dir):
    # set paths
    out_dir = "outputs_sub3"
    img_out_dir = "img_check_sub3/pts_check"
    vessel_model_dir = "model_sub3/vessel"
    localization_model_dir = "model_sub3/od_fovea_localization"
    utils_sub3.make_new_dir(out_dir)
    utils_sub3.make_new_dir(img_out_dir)
    
    # load the models and corresponding weights
    vessel_model = utils_sub3.load_network(vessel_model_dir)
    localization_model = utils_sub3.load_network(localization_model_dir)
    
    # make directories
    utils_sub3.make_new_dir(out_dir)
    utils_sub3.make_new_dir(img_out_dir)
        
    # iterate all images
    filepaths = utils_sub3.all_files_under(img_dir)
    image_no, od, fovea = [], {"x":[], "y":[]}, {"x":[], "y":[]}
    for filepath in filepaths:
        start_time = time.time()

        assert "IDRiD_" in os.path.basename(filepath)
        # load an img (tensor shape of [1,h,w,3])
        img = utils_sub3.imagefiles2arrs([filepath])
        _, h, w, _ = img.shape
        assert h == 2848 and w == 4288
        resized_img = utils_sub3.resize_img(img)
        
        # run inference
        vessel = vessel_model.predict(utils_sub3.normalize(resized_img, "vessel_segmentation"), batch_size=1)
        predicted_coords, _ = localization_model.predict([utils_sub3.normalize(resized_img, "od_from_fundus_vessel"), (vessel * 255).astype(np.uint8) / 255.], batch_size=1)
        
        # convert to coordinates in the original image
        coords = utils_sub3.convert_coords_to_original_scale(predicted_coords)
        
        # save the result
        utils_sub3.save_imgs_with_pts(img.astype(np.uint8), coords.astype(np.uint16), img_out_dir, os.path.basename(filepath).replace(".jpg", ""))
        od["y"].append(coords[0, 0])
        od["x"].append(coords[0, 1])
        fovea["y"].append(coords[0, 2])
        fovea["x"].append(coords[0, 3])
        image_no.append(os.path.basename(filepath).replace(".jpg", ""))
    
        print "duration: {} sec".format(time.time() - start_time)

    df_od = pd.DataFrame({'Image No':image_no, 'X-Coordinate':od["x"], 'Y-Coordinate':od["y"]})
    df_fovea = pd.DataFrame({'Image No':image_no, 'X-Coordinate':fovea["x"], 'Y-Coordinate':fovea["y"]})
    df_od.to_csv(os.path.join(out_dir, "VRT_OD localization.csv"), index=False)
    df_fovea.to_csv(os.path.join(out_dir, "VRT_fovea localization.csv"), index=False)

    
def segmentation_inference(img_dir):
    # set paths
    out_dir = "outputs_sub3/segmentation"
    vessel_model_dir = "model_sub3/vessel"
    segmentation_model_dir = "model_sub3/od_from_fundus_vessel"
    
    # load the models and corresponding weights
    vessel_model = utils_sub3.load_network(vessel_model_dir)
    od_from_fundus_vessel_model = utils_sub3.load_network(segmentation_model_dir)
    
    # make directories
    utils_sub3.make_new_dir(out_dir)
        
    # iterate all images
    filenames = utils_sub3.all_files_under(img_dir)
    for filename in filenames:
        assert "IDRiD_" in os.path.basename(filename)
        # load an img (tensor shape of [1,h,w,3])
        img = utils_sub3.imagefiles2arrs([filename])
        _, h, w, _ = img.shape
        assert h == 2848 and w == 4288
        resized_img = utils_sub3.resize_img(img)
        
        # run inference
        vessel = vessel_model.predict(utils_sub3.normalize(resized_img, "vessel_segmentation"), batch_size=1)
        od_seg, _ = od_from_fundus_vessel_model.predict([utils_sub3.normalize(resized_img, "od_from_fundus_vessel"), (vessel * 255).astype(np.uint8) / 255.], batch_size=1)
        
        # cut by threshold
        od_seg = od_seg[0, ..., 0]
        od_seg_zoomed_to_ori_scale = utils_sub3.sizeup(od_seg)
        od_seg_zoomed_to_ori_scale = utils_sub3.cut_by_threshold(od_seg_zoomed_to_ori_scale, threshold=0.44478172063827515)
        od_seg_zoomed_to_ori_scale_out = od_seg_zoomed_to_ori_scale * 255
        
        # save the result
        Image.fromarray((od_seg_zoomed_to_ori_scale_out).astype(np.uint8)).save(os.path.join(out_dir, os.path.basename(filename).replace("jpg","tif")))
    
