import utils
import argparse

# arrange arguments
parser = argparse.ArgumentParser()
FLAGS, _ = parser.parse_known_args()
gt_dir = "../data/original/OD_Segmentation_Training_Set/"
pred_dir = "../outputs/final_results_loss_weight_1_0.1_non_thre/od_from_fundus_vessel_visualization"

gt_filenames = utils.all_files_under(gt_dir)
pred_filenames = utils.all_files_under(pred_dir)

# build gt arrays
pred_all = utils.imagefiles2arrs(pred_filenames)
pred_all /= 255.
gt_all = utils.imagefiles2arrs(gt_filenames)
    
dice, threshold = utils.segmentation_optimal_threshold(gt_all, pred_all)

print "dice: {}".format(dice)
print "best threshold: {}".format(threshold)
