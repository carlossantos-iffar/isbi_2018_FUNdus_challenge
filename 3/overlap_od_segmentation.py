import utils
import numpy as np
import os

img_dir = "../data/original/Training_c/"
gt_dir = "../data/original/OD_Segmentation_Training_Set/"
pred_dir = "../outputs/final_results_segmentation_loss_weight_1_0.1/od_from_fundus_vessel/"
out_dir = "../outputs/od_seg_overlap_loss_weight_1_0.1"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    
img_filenames = utils.all_files_under(img_dir)
gt_filenames = utils.all_files_under(gt_dir)
pred_filenames = utils.all_files_under(pred_dir)

for index in range(len(gt_filenames)):
    # get gt and pred segs
    img = utils.imagefiles2arrs(img_filenames[index:index + 1]).astype(np.uint8)[0, ...]
    gt = utils.imagefiles2arrs(gt_filenames[index:index + 1]).astype(np.uint8)[0, ...]
    pred = utils.imagefiles2arrs(pred_filenames[index:index + 1]).astype(np.uint8)[0, ...]
    utils.compare_masks(img, gt, pred, out_dir, "{}.png".format(index))
