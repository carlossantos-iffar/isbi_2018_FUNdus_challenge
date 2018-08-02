import utils
import os
import numpy as np
import argparse

# arrange arguments
parser = argparse.ArgumentParser()
FLAGS, _ = parser.parse_known_args()
gt_dir_template = "../data/original/{}"
pred_dir_template = "../segmentation_results_for_sc2/{}"
tasks = ["EX"]
val_indices = [0, 1, 2, 3, 4, 54, 55, 56, 57, 58, 59, 60, 61]

for task in tasks:
    gt_dir = gt_dir_template.format(task)
    pred_dir = pred_dir_template.format(task)
    gt_filenames = utils.all_files_under(gt_dir)
    pred_filenames = utils.all_files_under(pred_dir)
    training_indices = ~np.in1d(range(len(pred_filenames)), val_indices)

    # build gt arrays
    index_gt = 0
    pred_all = utils.imagefiles2arrs(pred_filenames)
    gt_all = np.zeros(pred_all.shape)
    for index_pred in range(len(pred_filenames)):
        # build array of gt
        if index_gt < len(gt_filenames) and os.path.basename(pred_filenames[index_pred]).replace(".jpg", "") in os.path.basename(gt_filenames[index_gt]):
            gt = utils.imagefiles2arrs(gt_filenames[index_gt:index_gt + 1]).astype(np.uint8)[0, ...]
            gt_all[index_pred, ...] = gt
            index_gt += 1
        
    # compute sensitivity and specificity
    aupr_all, best_f1_all, best_f1_thresh_all, sen_all, ppv_all = utils.pr_metric(gt_all, pred_all)
    auroc_all = utils.AUC_ROC(gt_all, pred_all)
    aupr_training, best_f1_training, best_f1_thresh_training, sen_training, ppv_training = utils.pr_metric(gt_all[training_indices], pred_all[training_indices])
    auroc_training = utils.AUC_ROC(gt_all[training_indices], pred_all[training_indices])
    aupr_val, best_f1_val, best_f1_thresh_val, sen_val, ppv_val = utils.pr_metric(gt_all[val_indices], pred_all[val_indices])
    auroc_val = utils.AUC_ROC(gt_all[val_indices], pred_all[val_indices])
    
    # print results and store to lists
    sens = [sen_all, sen_training, sen_val]
    ppvs = [ppv_all, ppv_training, ppv_val]
    auprs = [aupr_all, aupr_training, aupr_val]
    aurocs = [auroc_all, auroc_training, auroc_val]
    best_f1s = [best_f1_all, best_f1_training, best_f1_val]
    best_f1_threshs = [best_f1_thresh_all, best_f1_thresh_training, best_f1_thresh_val]
    for index, eval_type in enumerate(["all", "traning", "val"]):
        print "**************** {} -- {} ****************".format(task, eval_type)
        print "sensitivity: {}".format(sens[index])
        print "ppv: {}".format(ppvs[index])
        print "f1: {}".format(best_f1s[index])
        print "best f1 threshold: {}".format(best_f1_threshs[index])
        print "aupr: {}".format(auprs[index])
        print "auroc: {}".format(aurocs[index])
