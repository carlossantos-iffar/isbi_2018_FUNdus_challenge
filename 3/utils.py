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
from scipy.misc import imresize
from scipy.ndimage import zoom

x_offset = 230
ori_h, ori_w, cropped_w = 2848, 4288, 3500
process_len = 640
y_offset = (cropped_w - ori_h) // 2


def segmentation_optimal_threshold(gt_all, pred_all):
    precision, recall, thresholds = precision_recall_curve(gt_all.flatten(), pred_all.flatten(), pos_label=1)
    best_dice = -1
    for index in range(len(precision)):
        curr_dice = 2.*precision[index] * recall[index] / (precision[index] + recall[index])
        if best_dice < curr_dice:
            best_dice = curr_dice
            best_threshold = thresholds[index]

    return best_dice, best_threshold
    

def compare_masks(img, true_mask, pred_mask, out_dir, fname):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_img = img * 0.5
    out_img[..., 1] += true_mask * 127
    out_img[..., 2] += pred_mask * 127
    Image.fromarray(out_img.astype(np.uint8)).save(os.path.join(out_dir, fname))


def seg_metrics(true, pred):
    cm = confusion_matrix(true.flatten(), pred.flatten())
    spe = 1.*cm[0, 0] / (cm[0, 1] + cm[0, 0])
    sen = 1.*cm[1, 1] / (cm[1, 0] + cm[1, 1])
    dice_val = 2.*cm[1, 1] / (cm[1, 0] + 2 * cm[1, 1] + cm[0, 1])
    jaccard_val = 1.*cm[1, 1] / (cm[1, 0] + cm[1, 1] + cm[0, 1])
    return cm, spe, sen, dice_val, jaccard_val


def dist(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def in_circle(pt, c_pt, r):
    return dist(pt, c_pt) <= r


def fovea_mask(coords):
    actmap_size, actmap_vessel_size = 40, 20
    assert len(coords.shape) == 2
    n = coords.shape[0]
    actmap = np.zeros((n, actmap_size, actmap_size, 1))
    actmap_vessel = np.zeros((n, actmap_vessel_size, actmap_vessel_size, 1))
    y, x = np.ogrid[:actmap_size, :actmap_size]
    y_vessel, x_vessel = np.ogrid[:actmap_vessel_size, :actmap_vessel_size]

    for i in range(n):
        h_c_actmap = int(coords[i, 0] * actmap_size)
        w_c_actmap = int(coords[i, 1] * actmap_size)
        h_c_actmap_vessel = int(coords[i, 0] * actmap_vessel_size)
        w_c_actmap_vessel = int(coords[i, 1] * actmap_vessel_size)
        actmap[i, in_circle((y, x), (h_c_actmap, w_c_actmap), 3), 0] = 1
        actmap_vessel[i, in_circle((y_vessel, x_vessel), (h_c_actmap_vessel, w_c_actmap_vessel), 1), 0] = 1

    return actmap, actmap_vessel


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


def resize_img(img):
    width_range = x_offset, x_offset + cropped_w
    img = img[:, :, width_range[0]:width_range[1], :]
    _, img_h, img_w, _ = img.shape
    len_side = width_range[1] - width_range[0]
    padded = np.zeros((len_side, len_side, 3))
    padded[(len_side - img_h) // 2:(len_side - img_h) // 2 + img_h, (len_side - img_w) // 2:(len_side - img_w) // 2 + img_w, ...] = img[0, ...]
    resized_img = imresize(padded, (process_len, process_len), 'bilinear')
    
    return np.expand_dims(resized_img, axis=0)


def sizeup(seg_result):
    final_result = np.zeros((ori_h, ori_w))
    upscale_ratio = 1.*cropped_w / process_len
    od_seg_scaled_up = zoom(seg_result, upscale_ratio, order=1)
    final_result[:, x_offset:x_offset + cropped_w] = od_seg_scaled_up[y_offset:y_offset + ori_h, :]
    return final_result


def convert_coords_to_original_scale(coords):
    coords_ori = np.zeros(coords.shape)

    for index in range(coords_ori.shape[0]):
        od_y, od_x, fovea_y, fovea_x = coords[index, ...]
        
        od_y *= cropped_w
        od_x *= cropped_w
        fovea_y *= cropped_w
        fovea_x *= cropped_w
        od_y -= y_offset
        fovea_y -= y_offset
        od_x += x_offset
        fovea_x += x_offset
        
        coords_ori[index, ...] = int(od_y), int(od_x), int(fovea_y), int(fovea_x)
    
    return coords_ori


def cut_by_threshold(segmented, threshold):
    thresholded = np.copy(segmented)
    thresholded[segmented > threshold] = 1
    thresholded[segmented <= threshold] = 0
    return thresholded


def normalize(img, task):
    assert len(img.shape) == 4 and img.shape[0] == 1
    new_img = np.zeros(img.shape)
    if task == "vessel_segmentation":
        # z score with mean, std (batchsize=1)
        mean = np.mean(img[0, ...][img[0, ..., 0] > 40.0], axis=0)
        std = np.std(img[0, ...][img[0, ..., 0] > 40.0], axis=0)
        new_img[0, ...] = (img[0, ...] - mean) / std
    elif task == "od_from_fundus_vessel" or task == "od_in_original_scale":
        means, stds = np.zeros(3), np.array([255.0, 255.0, 255.0])
        for i in range(3):
            if len(img[0, ..., i][img[0, ..., i] > 10]) > 1:
                means[i] = np.mean(img[0, ..., i][img[0, ..., i] > 10])
                std_val = np.std(img[0, ..., i][img[0, ..., i] > 10])
                stds[i] = std_val if std_val > 0 else 255.0
        new_img[0, ...] = (img[0, ...] - means) / stds
    return new_img


def transform_coords(df_od, df_fv):
    # compensate for background crop & padding & normalize to [0,1]
    ratio = 1. / cropped_w
    df_od.loc[:, "filename"] = df_od.loc[:, "Image No"]
    df_od.loc[:, "y_disc"] = (df_od.loc[:, "Y - Coordinate"] + y_offset) * ratio   
    df_od.loc[:, "x_disc"] = (df_od.loc[:, "X- Coordinate"] - x_offset) * ratio  
    df_od.loc[:, "y_fovea"] = (df_fv.loc[:, "Y - Coordinate"] + y_offset) * ratio   
    df_od.loc[:, "x_fovea"] = (df_fv.loc[:, "X- Coordinate"] - x_offset) * ratio
    
    return df_od[["filename", "y_disc", "x_disc", "y_fovea", "x_fovea"]]


def load_coords(fundus_dir, vessel_dir, od_label_path, fovea_label_path):
    fundus_fns = np.array(all_files_under(fundus_dir, append_path=False)) 
    vessel_fns = np.array(all_files_under(vessel_dir, append_path=False))
    assert (fundus_fns == vessel_fns).all()
    
    # get points
    df_od = pd.read_csv(od_label_path)
    df_fv = pd.read_csv(fovea_label_path)
    df = transform_coords(df_od, df_fv)
    pts = zip(df.y_disc, df.x_disc, df.y_fovea, df.x_fovea)
    label_dict = dict(zip(df.filename, pts))
    
    new_fundus_filepaths = []
    new_vessel_filepaths = []
    coord_list = []
    for index in range(len(fundus_fns)):
        fname = fundus_fns[index].replace(".tif", "")
        if fname in label_dict:
            coord_list.append(label_dict[fname])
            new_fundus_filepaths.append(os.path.join(fundus_dir, fundus_fns[index]))
            new_vessel_filepaths.append(os.path.join(vessel_dir, fundus_fns[index]))

    return np.array(new_fundus_filepaths), np.array(new_vessel_filepaths), np.array(coord_list)


def save_imgs_with_pts(imgs, pts, out_dir, uid):
    if imgs.dtype != np.uint8:
        imgs = (10 * imgs + 100).astype(np.uint8)
    for index in range(imgs.shape[0]):
        if imgs.dtype != np.uint8:
            imgs[index, int(imgs.shape[1] * pts[index, 0]) - 3:int(imgs.shape[1] * pts[index, 0]) + 3,
                        int(imgs.shape[2] * pts[index, 1]) - 3:int(imgs.shape[2] * pts[index, 1]) + 3, :] = (0, 0, 255)
            imgs[index, int(imgs.shape[1] * pts[index, 2]) - 3:int(imgs.shape[1] * pts[index, 2]) + 3,
                        int(imgs.shape[2] * pts[index, 3]) - 3:int(imgs.shape[2] * pts[index, 3]) + 3, :] = (0, 0, 255)
        else:
            imgs[index, pts[index, 0] - 10:pts[index, 0] + 10, pts[index, 1] - 10: pts[index, 1] + 10, :] = (0, 0, 255)
            imgs[index, pts[index, 2] - 10:pts[index, 2] + 10, pts[index, 3] - 10: pts[index, 3] + 10, :] = (0, 0, 255)
            
        Image.fromarray(imgs[index, :, :].astype(np.uint8)).save(os.path.join(out_dir, str(uid) + "_{}.png".format(index)))


def save_imgs_with_pts_fovea(imgs, pts, out_dir, uid):
    if imgs.dtype != np.uint8:
        imgs = (10 * imgs + 100).astype(np.uint8)
    for index in range(imgs.shape[0]):
        if pts.dtype != np.uint8:
            imgs[index, int(imgs.shape[1] * pts[index, 0]) - 3:int(imgs.shape[1] * pts[index, 0]) + 3,
                        int(imgs.shape[2] * pts[index, 1]) - 3:int(imgs.shape[2] * pts[index, 1]) + 3, :] = (0, 0, 255)
        else:
            imgs[index, pts[index, 0] - 10:pts[index, 0] + 10, pts[index, 1] - 10: pts[index, 1] + 10, :] = (0, 0, 255)
            
        Image.fromarray(imgs[index, :, :].astype(np.uint8)).save(os.path.join(out_dir, str(uid) + "_{}.png".format(index)))


def save_vessels_with_pts(vessels, pts, out_dir, uid):
    imgs = (255 * vessels).astype(np.uint8)
    for index in range(imgs.shape[0]):
        imgs[index, int(imgs.shape[1] * pts[index, 0]) - 10:int(imgs.shape[1] * pts[index, 0]) + 10,
                    int(imgs.shape[2] * pts[index, 1]) - 10:int(imgs.shape[2] * pts[index, 1]) + 10] = 255
        imgs[index, int(imgs.shape[1] * pts[index, 2]) - 10:int(imgs.shape[1] * pts[index, 2]) + 10,
                    int(imgs.shape[2] * pts[index, 3]) - 10:int(imgs.shape[2] * pts[index, 3]) + 10] = 255
        Image.fromarray(imgs[index, :, :].astype(np.uint8)).save(os.path.join(out_dir, str(uid) + "_{}.png".format(index)))


def load_augmented_lm(fundus, vessel, lm, augment=True):
    # read image file
    fundus = imagefiles2arrs([fundus], augment, z_score=True)[0, ...]
    vessel = np.expand_dims(imagefiles2arrs([vessel], augment=False, z_score=False)[0, ...], axis=2) / 255
    assert len(fundus.shape) == 3 and len(vessel.shape) == 3
    # random flip and rotation
    if augment:
        if random.getrandbits(1):
            fundus = fundus[:, ::-1, :]  # flip an image
            vessel = vessel[:, ::-1, :]  
            lm[1] = 1 - lm[1]  # x_disc
            lm[3] = 1 - lm[3]  # x_fovea  
        r_angle = random.randint(0, 359)
        lm[:2] = rotate_pt((0.5, 0.5), (lm[0], lm[1]), r_angle)
        lm[2:] = rotate_pt((0.5, 0.5), (lm[2], lm[3]), r_angle)
        fundus = rotate(fundus, r_angle, axes=(0, 1), reshape=False)
        vessel = rotate(vessel, r_angle, axes=(0, 1), reshape=False)
    return fundus, vessel, lm


def od_fovea_maps(img_shape, coords):
    h, w = img_shape
    n = coords.shape[0]
    od_maps = np.zeros((n, h, w, 1))
    fovea_maps = np.zeros((n, h, w, 1))
    for i in range(n):
        od_y, od_x, fv_y, fv_x = coords[i, ...]
        od_maps[i, int(h * od_y) - 1:int(h * od_y) + 1, int(w * od_x) - 1:int(w * od_x) + 1, 0] = 1
        fovea_maps[i, int(h * fv_y) - 1:int(h * fv_y) + 1, int(w * fv_x) - 1:int(w * fv_x) + 1, 0] = 1
    return od_maps, fovea_maps


def mean_Euclidean(coords_list, generated_list):
    sq_diff = np.square(generated_list - coords_list)
    disc_rmse_list = np.sqrt(np.sum(sq_diff[:, :2], axis=1))
    fovea_rmse_list = np.sqrt(np.sum(sq_diff[:, 2:], axis=1))
    return np.mean(disc_rmse_list), np.mean(fovea_rmse_list)


def mean_Euclidean_fovea(coords_list, generated_list):
    sq_diff = np.square(generated_list - coords_list)
    rmse_list = np.sqrt(np.sum(sq_diff, axis=1))
    return np.mean(rmse_list)


def check_input_localization(imgs, vessels, od_maps, fovea_maps, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for i in range(imgs.shape[0]):
        out_img = (10 * imgs[i, ...] + 100).astype(np.uint8)
        superimposed = np.copy(out_img)
        if out_img.shape[2] == 3:  # in case of rgb
            superimposed[..., 2] = out_img[..., 2] + vessels[i, ..., 0] * 0.5 * 255
        else:  # in case of gray scale
            out_img = np.squeeze(out_img, axis=2)
            superimposed = out_img + vessels[i, ..., 0] * 0.5 * 255
        Image.fromarray(out_img.astype(np.uint8)).save(os.path.join(out_dir, "input_{}.png".format(i + 1)))
        Image.fromarray((vessels[i, ..., 0] * 255).astype(np.uint8)).save(os.path.join(out_dir, "vessel_{}.png".format(i + 1)))
        Image.fromarray((od_maps[i, ..., 0] * 255).astype(np.uint8)).save(os.path.join(out_dir, "od_{}.png".format(i + 1)))
        Image.fromarray((fovea_maps[i, ..., 0] * 255).astype(np.uint8)).save(os.path.join(out_dir, "fovea_{}.png".format(i + 1)))


def check_input(imgs, masks, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for i in range(imgs.shape[0]):
        out_img = (30 * imgs[i, ...] + 100).astype(np.uint8)
        superimposed = np.copy(out_img)
        if out_img.shape[2] == 3:  # in case of rgb
            superimposed[..., 2] = out_img[..., 2] + masks[i, ..., 0] * 0.5 * 255
        else:  # in case of gray scale
            out_img = np.squeeze(out_img, axis=2)
            superimposed = out_img + masks[i, ..., 0] * 0.5 * 255
        Image.fromarray(out_img.astype(np.uint8)).save(os.path.join(out_dir, "input_{}.png".format(i + 1)))
        Image.fromarray(superimposed.astype(np.uint8)).save(os.path.join(out_dir, "superimposed_{}.png".format(i + 1)))
        Image.fromarray((masks[i, ..., 0] * 255).astype(np.uint8)).save(os.path.join(out_dir, "mask_{}.png".format(i + 1)))


def save_fig_fovea_segmentation(imgs, masks, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for i in range(imgs.shape[0]):
        out_img = (30 * imgs[i, ...] + 100).astype(np.uint8)
        superimposed = np.copy(out_img)
        if out_img.shape[2] == 3:  # in case of rgb
            superimposed[..., 2] = out_img[..., 2] + masks[i, ..., 0] * 0.5 * 255
        else:  # in case of gray scale
            out_img = np.squeeze(out_img, axis=2)
            superimposed = out_img + masks[i, ..., 0] * 0.5 * 255
        Image.fromarray(out_img.astype(np.uint8)).save(os.path.join(out_dir, "input_{}.png".format(i + 1)))
        Image.fromarray((masks[i, ..., 0] * 255).astype(np.uint8)).save(os.path.join(out_dir, "mask_{}.png".format(i + 1)))


# helper functions
def intersects(self, other):
    return not (self[1][1] < other[0][1] or 
                self[0][1] > other[1][1] or
                self[1][0] < other[0][0] or
                self[0][0] > other[1][0])


def coord_image(coord, img_size):
    # coord : x,y
    # img_size : h,w
    return np.array([np.clip(coord[0], 0, img_size[1]), np.clip(coord[1], 0, img_size[0])])


def add_bright_blob(fundus, seg):
    img_h, img_w, _ = fundus.shape
    
    # hyperparameters
    pos_bb_margin = 0.5
    n_aug_blobs = np.random.randint(0, 6)
    
    # get bounding box of od, format : [(top_left.x,top_left.y), (bottom_right.x, bottom_right.y)]
    # mask image has values of 0 and positive value
    labels, n_labels = measure.label(seg > 200, return_num=True)
    assert n_labels == 1
    # get coordinates of top left and bottom right
    row_inds, col_inds = np.where(labels == 1)
    top_left = np.array((np.min(col_inds), np.min(row_inds)))
    bottom_right = np.array((np.max(col_inds), np.max(row_inds)))
    # enlarge by 'margin'
    center = (top_left + bottom_right) / 2
    top_left = (center - (center - top_left) * (1 + 1.*pos_bb_margin)).astype(np.int)
    bottom_right = (center + (bottom_right - center) * (1 + 1.*pos_bb_margin)).astype(np.int)
    # adjust if outbound
    top_left = coord_image(top_left, [img_h, img_w])
    bottom_right = coord_image(bottom_right, [img_h, img_w])
    bbox_pos = [top_left, bottom_right]
        
    # superimpose patch when not overlapped
    n_current = 0
    while n_current < n_aug_blobs:
        # augment od patch
        patch = np.copy(fundus[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
        r_angle = random.randint(0, 359)
        x_scale, y_scale = random.uniform(1. / 1.05, 1.05), random.uniform(1. / 1.05, 1.05)
        shear = random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
        x_translation, y_translation = random.randint(-10, 10), random.randint(-10, 10)
        tform = AffineTransform(scale=(x_scale, y_scale), shear=shear,
                                translation=(x_translation, y_translation))
        patch_warped = warp(patch, tform.inverse, output_shape=(patch.shape[0], patch.shape[1]))
        patch_warped *= 255
        patch = rotate(patch_warped, r_angle, axes=(0, 1), order=1, reshape=False)

        h, w, _ = patch.shape
        top_left_x = np.random.randint(0, img_w - w)
        top_left_y = np.random.randint(0, img_h - h)
        candi = [[top_left_x, top_left_y], [top_left_x + w, top_left_y + h]]
       
        # check intersection with positivie bounding boxes       
        if not intersects(candi, bbox_pos):
            fundus[top_left_y:top_left_y + h, top_left_x:top_left_x + w, :] = patch
            n_current += 1
    
    return fundus
        

def load_augmented_fundus_vessel(fundus_fnames, vessel_fnames, seg_fnames, augment):
    assert len(fundus_fnames) == len(vessel_fnames) and len(fundus_fnames) == len(seg_fnames)
    n_imgs = len(fundus_fnames)
    img_shape = image_shape(fundus_fnames[0]) 
    fundus_arr = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
    vessel_arr = np.zeros((n_imgs, img_shape[0], img_shape[1], 1))
    seg_arr = np.zeros((len(seg_fnames), img_shape[0], img_shape[1], 1))
        
    for file_index in xrange(n_imgs):
        fundus = np.array(Image.open(fundus_fnames[file_index]))
        vessel = np.array(Image.open(vessel_fnames[file_index]))
        seg = np.array(Image.open(seg_fnames[file_index]))
        
        if augment:
            # random addition of optic disc like structure 
            fundus = add_bright_blob(fundus, seg)
            
            # random flip
            if random.getrandbits(1):
                fundus = fundus[:, ::-1]
                vessel = vessel[:, ::-1]
                seg = seg[:, ::-1]
            
            # affine transform (translation, scale, shear, rotation)
            r_angle = random.randint(0, 359)
            x_scale, y_scale = random.uniform(1. / 1.05, 1.05), random.uniform(1. / 1.05, 1.05)
            shear = random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
            x_translation, y_translation = random.randint(-10, 10), random.randint(-10, 10)
            tform = AffineTransform(scale=(x_scale, y_scale), shear=shear,
                                    translation=(x_translation, y_translation))
            fundus_warped = warp(fundus, tform.inverse, output_shape=(fundus.shape[0], fundus.shape[1]))
            fundus_warped *= 255
            vessel_warped = warp(vessel, tform.inverse, output_shape=(vessel.shape[0], vessel.shape[1]))
            seg_warped = warp(seg, tform.inverse, output_shape=(seg.shape[0], seg.shape[1]))
            fundus = rotate(fundus_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            vessel = rotate(vessel_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            seg = rotate(seg_warped, r_angle, axes=(0, 1), order=1, reshape=False)

            # convert to z score per image
            means, stds = np.zeros(3), np.array([255.0, 255.0, 255.0])
            for i in range(3):
                if len(fundus[..., i][fundus[..., i] > 10]) > 1:
                    means[i] = np.mean(fundus[..., i][fundus[..., i] > 10])
                    std_val = np.std(fundus[..., i][fundus[..., i] > 10])
                    stds[i] = std_val if std_val > 0 else 255
            fundus_arr[file_index] = (fundus - means) / stds
            vessel_arr[file_index, ...] = np.round(np.expand_dims(vessel, axis=2))
            seg_arr[file_index, ...] = np.round(np.expand_dims(seg, axis=2))

    return fundus_arr, vessel_arr, seg_arr


def load_augmented_fundus_mask(fundus_fnames, seg_fnames, augment):
    assert len(fundus_fnames) == len(seg_fnames)
    n_imgs = len(fundus_fnames)
    img_shape = image_shape(fundus_fnames[0]) 
    fundus_arr = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
    seg_arr = np.zeros((len(seg_fnames), img_shape[0], img_shape[1], 1))
        
    for file_index in xrange(n_imgs):
        fundus = np.array(Image.open(fundus_fnames[file_index]))
        seg = 255 * np.array(Image.open(seg_fnames[file_index]))
        
        if augment:
            # random addition of optic disc like structure 
            # random flip
            if random.getrandbits(1):
                fundus = fundus[:, ::-1]
                seg = seg[:, ::-1]

            # affine transform (translation, scale, shear, rotation)
            r_angle = random.randint(0, 359)
            x_scale, y_scale = random.uniform(1. / 1.05, 1.05), random.uniform(1. / 1.05, 1.05)
            shear = random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
            x_translation, y_translation = random.randint(-10, 10), random.randint(-10, 10)
            tform = AffineTransform(scale=(x_scale, y_scale), shear=shear,
                                    translation=(x_translation, y_translation))
            fundus_warped = warp(fundus, tform.inverse, output_shape=(fundus.shape[0], fundus.shape[1]))
            fundus_warped *= 255
            seg_warped = warp(seg, tform.inverse, output_shape=(seg.shape[0], seg.shape[1]))
            fundus = rotate(fundus_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            seg = rotate(seg_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            # convert to z score per image
            means, stds = np.zeros(3), np.array([255.0, 255.0, 255.0])
            for i in range(3):
                if len(fundus[..., i][fundus[..., i] > 10]) > 1:
                    means[i] = np.mean(fundus[..., i][fundus[..., i] > 10])
                    std_val = np.std(fundus[..., i][fundus[..., i] > 10])
                    stds[i] = std_val if std_val > 0 else 255
            fundus_arr[file_index] = (fundus - means) / stds
            seg_arr[file_index, ...] = np.round(np.expand_dims(seg, axis=2))
            
    return fundus_arr, seg_arr


def load_augmented(img_fnames, seg_fnames, augment, isvessel):
    img_shape = image_shape(img_fnames[0]) 
    if isvessel:
        image_arr = np.zeros((len(img_fnames), img_shape[0], img_shape[1], 1))
    else:
        image_arr = np.zeros((len(img_fnames), img_shape[0], img_shape[1], 3))
    seg_arr = np.zeros((len(seg_fnames), img_shape[0], img_shape[1], 1))
        
    for file_index in xrange(len(img_fnames)):
        img = np.array(Image.open(img_fnames[file_index]))
        seg = np.array(Image.open(seg_fnames[file_index]))
        
        if augment:
            # random flip
            if random.getrandbits(1):
                img = img[:, ::-1]  # flip an image
                seg = seg[:, ::-1]  # flip an image
            
            # affine transform (translation, scale, shear, rotation)
            r_angle = random.randint(0, 359)
            x_scale, y_scale = random.uniform(1. / 1.05, 1.05), random.uniform(1. / 1.05, 1.05)
            shear = random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
            x_translation, y_translation = random.randint(-10, 10), random.randint(-10, 10)
            tform = AffineTransform(scale=(x_scale, y_scale), shear=shear,
                                    translation=(x_translation, y_translation))
            img_warped = warp(img, tform.inverse, output_shape=(img.shape[0], img.shape[1]))
            img_warped *= 255
            seg_warped = warp(seg, tform.inverse, output_shape=(seg.shape[0], seg.shape[1]))
            img = rotate(img_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            seg = rotate(seg_warped, r_angle, axes=(0, 1), order=1, reshape=False)
        
        if isvessel:
            image_arr[file_index] = np.expand_dims(img, axis=2) / 255
            seg_arr[file_index, ...] = np.round(np.expand_dims(seg, axis=2))
        else:
            # convert to z score per image
            means, stds = np.zeros(3), np.array([255.0, 255.0, 255.0])
            for i in range(3):
                if len(img[..., i][img[..., i] > 10]) > 1:
                    means[i] = np.mean(img[..., i][img[..., i] > 10])
                    std_val = np.std(img[..., i][img[..., i] > 10])
                    stds[i] = std_val if std_val > 0 else 255
            image_arr[file_index] = (img - means) / stds
            seg_arr[file_index, ...] = np.round(np.expand_dims(seg, axis=2))

    return image_arr, seg_arr


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


def crop_around_od(fundus_arr, mask_arr, target_size):
    assert len(fundus_arr.shape) == 4 and len(mask_arr.shape) == 4
    assert fundus_arr.shape[:3] == mask_arr.shape[:3]
    fundus_cropped_arr = np.zeros((fundus_arr.shape[0], target_size[0], target_size[1], 3))
    mask_cropped_arr = np.zeros((fundus_arr.shape[0], target_size[0], target_size[1], 1))
    
    for index in range(fundus_arr.shape[0]):
        center = center_blob(mask_arr[index, ..., 0])
        fundus_cropped_arr[index, ...] = crop_img(fundus_arr[index, ...], center, target_size)
        mask_cropped_arr[index, ...] = crop_img(mask_arr[index, ...], center, target_size)
        
    return fundus_cropped_arr, mask_cropped_arr


def set_validationset_fundus_mask(fundus_fnames, mask_fnames, img_size):
    assert len(fundus_fnames) == len(mask_fnames) 
    n_imgs = len(fundus_fnames)
    img_shape = np.array(Image.open(fundus_fnames[0])).shape
    fundus_arr = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
    mask_arr = np.zeros((len(mask_fnames), img_shape[0], img_shape[1], 1))
        
    for file_index in xrange(n_imgs):
        # convert to z score per fundus image
        fundus = np.array(Image.open(fundus_fnames[file_index])).astype(np.float32)
        means, stds = np.zeros(3), np.array([255.0, 255.0, 255.0])
        for i in range(3):
            if len(fundus[..., i][fundus[..., i] > 10]) > 1:
                means[i] = np.mean(fundus[..., i][fundus[..., i] > 10])
                std_val = np.std(fundus[..., i][fundus[..., i] > 10])
                stds[i] = std_val if std_val > 0 else 255
        fundus_arr[file_index] = (fundus - means) / stds
        
        # od segmentation
        mask = np.array(Image.open(mask_fnames[file_index]))
        mask[mask > 0] = 1
        mask_arr[file_index, ...] = np.expand_dims(mask, axis=2)

    cropped_fundus, cropped_mask = crop_around_od(fundus_arr, mask_arr, img_size)
    return cropped_fundus, cropped_mask


def set_validationset_fundus_vessel(fundus_fnames, vessel_fnames, seg_fnames):
    assert len(fundus_fnames) == len(vessel_fnames) and len(vessel_fnames) == len(seg_fnames) 
    n_imgs = len(fundus_fnames)
    img_shape = np.array(Image.open(vessel_fnames[0])).shape
    fundus_arr = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
    vessel_arr = np.zeros((n_imgs, img_shape[0], img_shape[1], 1))
    seg_arr = np.zeros((len(seg_fnames), img_shape[0], img_shape[1], 1))
        
    for file_index in xrange(n_imgs):
        # convert to z score per fundus image
        fundus = np.array(Image.open(fundus_fnames[file_index])).astype(np.float32)
        means, stds = np.zeros(3), np.array([255.0, 255.0, 255.0])
        for i in range(3):
            if len(fundus[..., i][fundus[..., i] > 10]) > 1:
                means[i] = np.mean(fundus[..., i][fundus[..., i] > 10])
                std_val = np.std(fundus[..., i][fundus[..., i] > 10])
                stds[i] = std_val if std_val > 0 else 255
        fundus_arr[file_index] = (fundus - means) / stds
        
        # vessel
        vessel = np.array(Image.open(vessel_fnames[file_index])).astype(np.float32)
        vessel /= 255
        vessel_arr[file_index] = np.expand_dims(vessel, axis=2) 
        
        # od segmentation
        seg = np.array(Image.open(seg_fnames[file_index]))
        seg[seg > 0] = 1
        seg_arr[file_index, ...] = np.expand_dims(seg, axis=2)

    return fundus_arr, vessel_arr, seg_arr


def set_validationset_vessel(img_fnames, seg_fnames):
    img_shape = np.array(Image.open(img_fnames[0])).shape
    image_arr = np.zeros((len(img_fnames), img_shape[0], img_shape[1], 1))
    seg_arr = np.zeros((len(seg_fnames), img_shape[0], img_shape[1], 1))
        
    for file_index in xrange(len(img_fnames)):
        img = np.array(Image.open(img_fnames[file_index])).astype(np.float32)
        img /= 255
        seg = np.array(Image.open(seg_fnames[file_index]))
        seg[seg > 0] = 1
        image_arr[file_index] = np.expand_dims(img, axis=2) 
        seg_arr[file_index, ...] = np.expand_dims(seg, axis=2)

    return image_arr, seg_arr


def set_validationset(img_fnames, seg_fnames):
    img_shape = np.array(Image.open(img_fnames[0])).shape
    image_arr = np.zeros((len(img_fnames), img_shape[0], img_shape[1], 3))
    seg_arr = np.zeros((len(seg_fnames), img_shape[0], img_shape[1], 1))
        
    for file_index in xrange(len(img_fnames)):
        img = np.array(Image.open(img_fnames[file_index])).astype(np.float32)
        seg = np.array(Image.open(seg_fnames[file_index]))
        seg[seg > 0] = 1
        
        # convert to z score per image
        means, stds = np.zeros(3), np.array([255.0, 255.0, 255.0])
        for i in range(3):
            if len(img[..., i][img[..., i] > 10]) > 1:
                means[i] = np.mean(img[..., i][img[..., i] > 10])
                std_val = np.std(img[..., i][img[..., i] > 10])
                stds[i] = std_val if std_val > 0 else 255
        image_arr[file_index] = (img - means) / stds
        seg_arr[file_index, ...] = np.expand_dims(seg, axis=2)

    return image_arr, seg_arr


def print_metrics(itr, **kargs):
    print "*** Epoch {}  ====> ".format(itr),
    for name, value in kargs.items():
        print ("{} : {}, ".format(name, value)),
    print ""
    sys.stdout.flush()


def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    """
    AUC_ROC = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    return AUC_ROC

    
def AUC_PR(true_vessel_img, pred_vessel_img):
    """
    Precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(), pos_label=1)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec


def dice_coeff(ytrue, ypred):
    ypred_max = np.argmax(ypred, axis=3)
    ytrue_max = np.argmax(ytrue, axis=3)
    ypred_pos = np.sum(ypred_max > 0)
    ytrue_pos = np.sum(ytrue_max > 0)
    
    y_intersect = np.sum((ypred_max == ytrue_max) & (ypred_max > 0) & (ytrue_max > 0))
    return 2.0 * y_intersect / (ypred_pos + ytrue_pos)


def rotate_pt(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    rad = angle * math.pi / 180
    oy, ox = origin
    py, px = point
    qx = ox + math.cos(rad) * (px - ox) + math.sin(rad) * (py - oy)
    qy = oy - math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)
    return qy, qx


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
