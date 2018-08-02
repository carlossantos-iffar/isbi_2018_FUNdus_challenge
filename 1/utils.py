import os
import sys

from PIL import Image, ImageEnhance
from scipy.ndimage import rotate
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure
from skimage.transform import warp, AffineTransform

import natsort as ns
import numpy as np
import random
import pandas as pd
import math
from scipy.ndimage import zoom
from sklearn.metrics.classification import confusion_matrix
from scipy.misc import imresize

x_offset = 230
ori_h, ori_w, cropped_w = 2848, 4288, 3500
process_len = 640
y_offset = (cropped_w - ori_h) // 2


def pr_metric(true_img, pred_img):
    """
    Precision-recall curve
    """
    precision, recall, thresholds = precision_recall_curve(true_img.flatten(), pred_img.flatten())
    AUC_prec_rec = auc(recall, precision)
    
    # compute bset f1
    best_f1 = -1
    for index in range(len(precision)):
        curr_f1 = 2.*precision[index] * recall[index] / (precision[index] + recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            prec = precision[index]
            sen = recall[index]
            best_threshold = thresholds[index]
    return AUC_prec_rec, best_f1, best_threshold, sen, prec


def sizeup(seg_result, process_len):
    final_result = np.zeros((ori_h, ori_w))
    upscale_ratio = 1.*cropped_w / process_len
    seg_scaled_up = zoom(seg_result, upscale_ratio, order=1)
    final_result[:, x_offset:x_offset + cropped_w] = seg_scaled_up[y_offset:y_offset + ori_h, :]
    return final_result


def pad_img(img, target_size):
    assert len(img.shape) == 3
    assert target_size[0] >= img.shape[0] and target_size[1] >= img.shape[1]
    if len(img.shape) == 3:
        img_h, img_w, img_d = img.shape
        padded = np.zeros((target_size[0], target_size[1], img_d))
        padded[(target_size[0] - img_h) // 2:(target_size[0] - img_h) // 2 + img_h, (target_size[1] - img_w) // 2:(target_size[1] - img_w) // 2 + img_w, ...] = img
    elif len(img.shape) == 2:
        img_h, img_w = img.shape
        padded = np.zeros((target_size[0], target_size[1]))
        padded[(target_size[0] - img_h) // 2:(target_size[0] - img_h) // 2 + img_h, (target_size[1] - img_w) // 2:(target_size[1] - img_w) // 2 + img_w] = img
        
    return padded


def background_foreground_weight(seg):
    seg_flattened = seg.flatten()
    n_foreground = len(seg_flattened[seg_flattened == 1])
    n_background = len(seg_flattened[seg_flattened == 0])
    weights = 1.*n_foreground / (n_foreground + n_background), 1.*n_background / (n_foreground + n_background) 
    print "class weight: {}".format(weights)
    return weights

    
def cut_by_threshold(segmented, threshold):
    thresholded = np.copy(segmented)
    thresholded[segmented > threshold] = 1
    thresholded[segmented <= threshold] = 0
    return thresholded


def normalize(img, method):
    assert len(img.shape) == 4 and img.shape[0] == 1

    if method == "rescale":
        new_img = np.zeros(img.shape)
        new_img[0, ...] = img[0, ...] / 255.
        return new_img
    elif method == "rescale_mean_subtract":
        new_img = np.zeros(img.shape)
        img = img / 255.
        means = np.zeros(3)
        for i in range(3):
            if len(img[0, ..., i][img[0, ..., i] > 10. / 255]) > 1:
                means[i] = np.mean(img[0, ..., i][img[0, ..., i] > 10. / 255])
        new_img[0, ...] = img[0, ...] - means   
        return new_img


def check_input(imgs, masks, out_dir):
#     print "img_val: {}".format(np.unique(imgs))
#     print "mask_val: {}".format(np.unique(masks))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for i in range(imgs.shape[0]):
        out_img = (255 * imgs[i, ...]).astype(np.uint8)
        superimposed = np.copy(out_img)
        if out_img.shape[2] == 3:  # in case of rgb
            superimposed[..., 2] = out_img[..., 2] + masks[i, ..., 0] * 0.5 * 255
        else:  # in case of gray scale
            out_img = np.squeeze(out_img, axis=2)
            superimposed = out_img + masks[i, ..., 0] * 0.5 * 255
        Image.fromarray(out_img.astype(np.uint8)).save(os.path.join(out_dir, "input_{}.png".format(i + 1)))
        Image.fromarray(superimposed.astype(np.uint8)).save(os.path.join(out_dir, "superimposed_{}.png".format(i + 1)))
        Image.fromarray((masks[i, ..., 0] * 255).astype(np.uint8)).save(os.path.join(out_dir, "mask_{}.png".format(i + 1)))


# helper functions
def intersects(self, other):
    return not (self[1][1] < other[0][1] or 
                self[0][1] > other[1][1] or
                self[1][0] < other[0][0] or
                self[0][0] > other[1][0])


def load_augmented_patches(fundus_batch, seg_batch, patch_size, augment, normalize):
    assert len(fundus_batch.shape) == 4 and len(seg_batch.shape) == 4
    assert fundus_batch.shape[0] == seg_batch.shape[0]
    
    n_imgs = fundus_batch.shape[0]
    fundus_batch_aug = np.zeros((n_imgs, patch_size[0], patch_size[1], 3))
    seg_batch_aug = np.zeros((n_imgs, patch_size[0], patch_size[1], 1))
    
    for index in range(n_imgs):
        fundus_batch_aug[index, ...], seg_batch_aug[index, ...] = load_augmented_patch(fundus_batch[index, ...], seg_batch[index, ...], patch_size, augment, normalize)
    
    return fundus_batch_aug, seg_batch_aug


def imagefiles2arrs(filenames, augment=False, z_score=False):
    img_shape = image_shape(filenames[0])
    if len(img_shape) == 3:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)
    # convert to z score per image
    for file_index in xrange(len(filenames)):
        im = Image.open(filenames[file_index])
        if augment:
            en = ImageEnhance.Color(im)
            im = en.enhance(random.uniform(0.9, 1.2))
        img = np.array(im)
        if z_score:
            means, stds = np.zeros(3), np.array([255.0, 255.0, 255.0])
            for i in range(3):
                if len(img[..., i][img[..., i] > 10]) > 1:
                    means[i] = np.mean(img[..., i][img[..., i] > 10])
                    std_val = np.std(img[..., i][img[..., i] > 10])
                    stds[i] = std_val if std_val > 0 else 255
            images_arr[file_index] = (img - means) / stds
        else:
            images_arr[file_index] = img
    return images_arr


def load_augmented_patch(fundus, seg, patch_size, augment, normalize):
    assert len(fundus.shape) == 3 and len(seg.shape) == 3
    
#     fundus, seg = crop_patches(fundus, seg, patch_size)
    
    if augment:
        # random addition of optic disc like structure 
        # random flip
        if random.getrandbits(1):
            fundus = fundus[:, ::-1, :]
            seg = seg[:, ::-1]

        # affine transform (translation, scale, shear, rotation)
        r_angle = random.randint(0, 359)
        x_scale, y_scale = random.uniform(1. / 1.05, 1.05), random.uniform(1. / 1.05, 1.05)
        shear = random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
        x_translation, y_translation = random.randint(-10, 10), random.randint(-10, 10)
        tform = AffineTransform(scale=(x_scale, y_scale), shear=shear,
                                translation=(x_translation, y_translation))
        fundus_warped = warp(fundus, tform.inverse, output_shape=(fundus.shape[0], fundus.shape[1]))
        seg_warped = warp(seg, tform.inverse, output_shape=(seg.shape[0], seg.shape[1]))
        fundus = rotate(fundus_warped, r_angle, axes=(0, 1), order=1, reshape=False)
        fundus = fundus * (1 + random.random() * 0.1) if random.getrandbits(1) else fundus * (1 - random.random() * 0.1)
        fundus = np.clip(fundus, 0, 255)
        seg = rotate(seg_warped, r_angle, axes=(0, 1), order=1, reshape=False)
    
    if normalize == "rescale_mean_subtract":
        means = np.zeros(3)
        for i in range(3):
            if len(fundus[..., i][fundus[..., i] > 10. / 255]) > 1:
                means[i] = np.mean(fundus[..., i][fundus[..., i] > 10. / 255])
        fundus -= means
    
#     means, stds = np.zeros(3), np.array([255.0, 255.0, 255.0])
#     for i in range(3):
#         if len(fundus[..., i][fundus[..., i] > 10]) > 1:
#             means[i] = np.mean(fundus[..., i][fundus[..., i] > 10])
#             std_val = np.std(fundus[..., i][fundus[..., i] > 10])
#             stds[i] = std_val if std_val > 0 else 255
#     fundus = (fundus - means) / stds
    
    return fundus, np.round(seg)


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]
    
    if sort:
        filenames = ns.natsorted(filenames)
    
    return filenames


def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def center_blob(mask):
    gt_labels = measure.label(mask > 0, return_num=False)
    gt_row_inds, gt_col_inds = np.where(gt_labels == 1)
    h_center = (np.min(gt_row_inds) + np.max(gt_row_inds)) // 2
    w_center = (np.min(gt_col_inds) + np.max(gt_col_inds)) // 2
    return h_center, w_center


def patch_in_original_scale(patch, ori_shape, center):
    mask_in_original_scale = np.zeros(ori_shape)
    patch_shape = patch.shape
    
    h_start_ori, h_end_ori = max(0, center[0] - patch_shape[0] // 2), min(ori_shape[0], center[0] + patch_shape[0] // 2)
    w_start_ori, w_end_ori = max(0, center[1] - patch_shape[1] // 2), min(ori_shape[1], center[1] + patch_shape[1] // 2)

    h_start_patch, h_end_patch = max(0, patch_shape[0] // 2 - center[0]), patch_shape[0] // 2 + min(patch_shape[0] // 2, ori_shape[0] - center[0])
    w_start_patch, w_end_patch = max(0, patch_shape[1] // 2 - center[1]), patch_shape[1] // 2 + min(patch_shape[1] // 2, ori_shape[1] - center[1])
    
    mask_in_original_scale[h_start_ori:h_end_ori, w_start_ori:w_end_ori, ...] = patch[h_start_patch:h_end_patch, w_start_patch:w_end_patch, ...]
    return mask_in_original_scale


def crop_img(img, center, target_size):
    h_center, w_center = center[0], center[1]
    half_h, half_w = target_size[0] // 2, target_size[1] // 2 

    h_start, h_end = max(0, h_center - half_h), min(img.shape[0], h_center + half_h)
    w_start, w_end = max(0, w_center - half_w), min(img.shape[1], w_center + half_w)
    cropped = img[h_start:h_end, w_start:w_end, :]
    padded = pad_img(cropped, target_size)
    return padded


def load_img(fn):
    return np.array(Image.open(fn)).astype(np.float32)


def load_fundus_mask(fundus_fns, mask_fns=None):
    img_shape = image_shape(fundus_fns[0])
    n_imgs = len(fundus_fns)
    fundus_arr = np.zeros((n_imgs,) + (img_shape[0], img_shape[1]) + (3,))
    mask_arr = np.zeros((n_imgs,) + (img_shape[0], img_shape[1]) + (1,))
    labels = np.zeros(n_imgs)
    if mask_fns:
        index_mask = 0
        for index in range(n_imgs):
            fundus_arr[index, ...] = load_img(fundus_fns[index]) / 255.
            if os.path.basename(fundus_fns[index]).replace(".tif", "") in os.path.basename(mask_fns[index_mask]):
                labels[index] = 1
                mask_arr[index, ..., 0] = load_img(mask_fns[index_mask])
                index_mask += 1
    else:
        for index in range(n_imgs):
            fundus_arr[index, ...] = load_img(fundus_fns[index]) / 255.
            
    return fundus_arr, mask_arr, labels


def metrics(gt_masks, pred_masks):
    cm = confusion_matrix(gt_masks.flatten(), np.round(pred_masks.flatten()))
    spe = 1.*cm[0, 0] / (cm[0, 1] + cm[0, 0]) if cm[0, 1] + cm[0, 0] != 0 else 0
    sen = 1.*cm[1, 1] / (cm[1, 0] + cm[1, 1]) if cm[1, 0] + cm[1, 1] != 0 else 0
    ppv = 1.*cm[1, 1] / (cm[0, 1] + cm[1, 1]) if cm[0, 1] + cm[1, 1] != 0 else 0
    return spe, sen, ppv


def crop_patches(fundus, mask, patch_size):
    patch_h, patch_w = patch_size
    
    # set left-top coordinates
    y = random.randint(0, ori_h - patch_h)
    x = random.randint(0, cropped_w - patch_w)
    
    fundus_patches = fundus[y:y + patch_h, x:x + patch_w, ...]
    mask_patches = mask[y:y + patch_h, x:x + patch_w, ...]
    
    return fundus_patches, mask_patches


def resize_img(img, target_len):
    width_range = x_offset, x_offset + cropped_w
    img = img[:, :, width_range[0]:width_range[1], :]
    _, img_h, img_w, _ = img.shape
    len_side = width_range[1] - width_range[0]
    padded = np.zeros((len_side, len_side, 3))
    padded[(len_side - img_h) // 2:(len_side - img_h) // 2 + img_h, (len_side - img_w) // 2:(len_side - img_w) // 2 + img_w, ...] = img[0, ...]
    resized_img = imresize(padded, (target_len, target_len), 'bicubic')
    
    return np.expand_dims(resized_img, axis=0)


def split_dataset(AR_path, NAR_path, seg_path, val_ratio, exclude_NAR=False):
    AR_fns, NAR_fns, seg_fns = all_files_under(AR_path), all_files_under(NAR_path), all_files_under(seg_path)
    n_AR, n_NAR = len(AR_fns), len(NAR_fns)
    
    fundus_AR, mask_AR, labels_AR = load_fundus_mask(AR_fns, seg_fns)
    n_val_AR = int(n_AR * val_ratio)
    
    if exclude_NAR:  # in case of SE
        train_fundus = fundus_AR[:-n_val_AR]
        train_mask = mask_AR[:-n_val_AR]
        train_labels = labels_AR[:-n_val_AR]
        val_fundus = fundus_AR[-n_val_AR:]
        val_mask = mask_AR[-n_val_AR:]
        val_labels = labels_AR[-n_val_AR:]
        return (train_fundus, train_mask, train_labels), (val_fundus, val_mask, val_labels) 
    else:
        fundus_NAR, mask_NAR, labels_NAR = load_fundus_mask(NAR_fns)
        n_val_NAR = int(n_NAR * val_ratio)
        train_fundus = np.concatenate([fundus_AR[n_val_AR:], fundus_NAR[n_val_NAR:]], axis=0)
        train_mask = np.concatenate([mask_AR[n_val_AR:], mask_NAR[n_val_NAR:]], axis=0)
        train_labels = np.concatenate([labels_AR[n_val_AR:], labels_NAR[n_val_NAR:]], axis=0)
        val_fundus = np.concatenate([fundus_AR[:n_val_AR], fundus_NAR[:n_val_NAR]], axis=0)
        val_mask = np.concatenate([mask_AR[:n_val_AR], mask_NAR[:n_val_NAR]], axis=0)
        val_labels = np.concatenate([labels_AR[:n_val_AR], labels_NAR[:n_val_NAR]], axis=0)
        return (train_fundus, train_mask, train_labels), (val_fundus, val_mask, val_labels) 


def print_metrics(itr, **kargs):
    print "*** Epoch {}  ====> ".format(itr),
    for name, value in kargs.items():
        print ("{} : {}, ".format(name, value)),
    print ""
    sys.stdout.flush()


def AUC_ROC(true_arr, pred_arr):
    """
    Area under the ROC curve with x axis flipped
    """
    AUC_ROC = roc_auc_score(true_arr.flatten(), pred_arr.flatten())
    return AUC_ROC

    
def AUC_PR(true_img, pred_img):
    """
    Precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(true_img.flatten(), pred_img.flatten(), pos_label=1)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec


def dice_coeff(ytrue, ypred):
    ypred_max = np.argmax(ypred, axis=3)
    ytrue_max = np.argmax(ytrue, axis=3)
    ypred_pos = np.sum(ypred_max > 0)
    ytrue_pos = np.sum(ytrue_max > 0)
    
    y_intersect = np.sum((ypred_max == ytrue_max) & (ypred_max > 0) & (ytrue_max > 0))
    return 2.0 * y_intersect / (ypred_pos + ytrue_pos)


class Scheduler:

    def __init__(self, schedules):
        self.schedules = schedules
        self.lr = schedules['lr']['0']

    def get_lr(self):
        return self.lr
        
    def update_steps(self, n_round):
        key = str(n_round)
        if key in self.schedules['lr']:
            self.lr = self.schedules['lr'][key]
