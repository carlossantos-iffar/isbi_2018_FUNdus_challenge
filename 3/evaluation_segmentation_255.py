import utils
import os
import numpy as np
from sklearn.metrics.classification import confusion_matrix
import argparse
from skimage import measure
from scipy.spatial.distance import dice

gt_dir = "../data/original/OD_Segmentation_Training_Set/"
pred_dir = "../../inference_codes/outputs_sub3/segmentation/"
gt_filenames = utils.all_files_under(gt_dir)
pred_filenames = utils.all_files_under(pred_dir)
list_sen, list_spe, list_jaccard, list_dice, list_dice_ori = [], [], [], [], []
for index in range(len(gt_filenames)):
    # get gt and pred segs
    gt = utils.imagefiles2arrs(gt_filenames[index:index + 1]).astype(np.uint8)[0, ...]
    pred = utils.imagefiles2arrs(pred_filenames[index:index + 1]).astype(np.uint8)[0, ...] // 255
    assert len(gt.shape) == 2 and len(pred.shape) == 2
    
    # compute sensitivity and specificity
    print pred_filenames[index]
    cm, spe, sen, dice_val, jaccard_val = utils.seg_metrics(gt, pred)
    
    # print results and store to lists
    print "--- {} ---".format(os.path.basename(gt_filenames[index]))
    print "specificity: {}".format(spe)
    print "sensitivity: {}".format(sen)
    print "jaccard: {}".format(jaccard_val)
    print "dice: {}".format(dice_val)
    list_spe.append(spe)
    list_sen.append(sen)
    list_jaccard.append(jaccard_val)
    list_dice.append(dice_val)
    list_dice_ori.append(dice_val)

# print all results & filenames
print "mean_sen: {}, min_sen: {}, max_sen: {}".format(np.mean(list_sen), np.min(list_sen), np.max(list_sen))
print "min_sen_file: {}, max_sen_file: {}".format(os.path.basename(gt_filenames[np.argmin(list_sen)]), os.path.basename(gt_filenames[np.argmax(list_sen)]))
print "mean_spe: {}, min_spe: {}, max_spe: {}".format(np.mean(list_spe), np.min(list_spe), np.max(list_spe))
print "min_spe_file: {}, max_spe_file: {}".format(os.path.basename(gt_filenames[np.argmin(list_spe)]), os.path.basename(gt_filenames[np.argmax(list_spe)]))
print "mean_jaccard: {}, min_jaccard: {}, max_jaccard: {}".format(np.mean(list_jaccard), np.min(list_jaccard), np.max(list_jaccard))
print "min_jaccard_file: {}, max_jaccard_file: {}".format(os.path.basename(gt_filenames[np.argmin(list_jaccard)]), os.path.basename(gt_filenames[np.argmax(list_jaccard)]))
print "mean_dice: {}, min_dice: {}, max_dice: {}".format(np.mean(list_dice), np.min(list_dice), np.max(list_dice))
print "min_dice_file: {}, max_dice_file: {}".format(os.path.basename(gt_filenames[np.argmin(list_dice)]), os.path.basename(gt_filenames[np.argmax(list_dice)]))
print "mean_dice_ori: {}, min_dice_ori: {}, max_dice_ori: {}".format(np.mean(list_dice_ori), np.min(list_dice_ori), np.max(list_dice_ori))
print "min_dice_ori_file: {}, max_f1_file: {}".format(os.path.basename(gt_filenames[np.argmin(list_dice_ori)]), os.path.basename(gt_filenames[np.argmax(list_dice_ori)]))

print "results for validation"
print "mean_sen: {}, min_sen: {}, max_sen: {}".format(np.mean(list_sen[:12]), np.min(list_sen[:12]), np.max(list_sen[:12]))
print "mean_spe: {}, min_spe: {}, max_spe: {}".format(np.mean(list_spe[:12]), np.min(list_spe[:12]), np.max(list_spe[:12]))
print "mean_jaccard: {}, min_jaccard: {}, max_jaccard: {}".format(np.mean(list_jaccard[:12]), np.min(list_jaccard[:12]), np.max(list_jaccard[:12]))
print "mean_dice: {}, min_dice: {}, max_dice: {}".format(np.mean(list_dice[:12]), np.min(list_dice[:12]), np.max(list_dice[:12]))
print "mean_dice_ori: {}, min_dice_ori: {}, max_dice_ori: {}".format(np.mean(list_dice_ori[:12]), np.min(list_dice_ori[:12]), np.max(list_dice_ori[:12]))

print "results for training"
print "mean_sen: {}, min_sen: {}, max_sen: {}".format(np.mean(list_sen[12:]), np.min(list_sen[12:]), np.max(list_sen[12:]))
print "mean_spe: {}, min_spe: {}, max_spe: {}".format(np.mean(list_spe[12:]), np.min(list_spe[12:]), np.max(list_spe[12:]))
print "mean_jaccard: {}, min_jaccard: {}, max_jaccard: {}".format(np.mean(list_jaccard[12:]), np.min(list_jaccard[12:]), np.max(list_jaccard[12:]))
print "mean_dice: {}, min_dice: {}, max_dice: {}".format(np.mean(list_dice[12:]), np.min(list_dice[12:]), np.max(list_dice[12:]))
print "mean_dice_ori: {}, min_dice_ori: {}, max_dice_ori: {}".format(np.mean(list_dice_ori[12:]), np.min(list_dice_ori[12:]), np.max(list_dice_ori[12:]))

