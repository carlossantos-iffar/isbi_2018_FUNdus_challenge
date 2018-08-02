import os
import sys

from PIL import Image, ImageEnhance
from scipy.ndimage import rotate
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure
from skimage.transform import warp, AffineTransform
from subprocess import Popen, PIPE

import natsort as ns
import numpy as np
import random
import pandas as pd
import math
from scipy.misc import imresize
from scipy.ndimage import zoom
from keras.models import model_from_json

x_offset = 230
ori_h, ori_w, cropped_w = 2848, 4288, 3500
process_len = 640
y_offset = (cropped_w - ori_h) // 2


def make_new_dir(out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


def load_network(dir_name):
    network_file = all_files_under(dir_name, extension=".json")
    weight_file = all_files_under(dir_name, extension=".h5")
    assert len(network_file) == 1 and len(weight_file) == 1
    with open(network_file[0], 'r') as f:
        network = model_from_json(f.read())
    network.load_weights(weight_file[0])
    network.trainable = False
    for l in network.layers:
        l.trainable = False
    return network


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


def check_input(imgs_mean_subt, imgs_z, vessels, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for i in range(imgs_mean_subt.shape[0]):
        out_img_mean_subt = (30 * imgs_mean_subt[i, ...] + 100).astype(np.uint8)
        out_img_z = (30 * imgs_z[i, ...] + 100).astype(np.uint8)
        out_vessel = (255 * vessels[i, ..., 0]).astype(np.uint8)
        Image.fromarray(out_img_mean_subt).save(os.path.join(out_dir, "fundus_mean_subt_{}.png".format(i + 1)))
        Image.fromarray(out_img_z).save(os.path.join(out_dir, "fundus_z_{}.png".format(i + 1)))
        Image.fromarray(out_vessel).save(os.path.join(out_dir, "vessel_{}.png".format(i + 1)))


def split(fundus_dir, vessel_dir, grade_path, grade_type, val_ratio):
    fundus_fns = np.array(all_files_under(fundus_dir, append_path=False)) 
    vessel_fns = np.array(all_files_under(vessel_dir, append_path=False))
    assert (fundus_fns == vessel_fns).all()

    n_data = len(fundus_fns)
    n_val = int(n_data * val_ratio)
    
    train_fundus_fns, train_vessel_fns, train_grades = load_fnames_grade(fundus_fns[n_val:], vessel_fns[n_val:], fundus_dir, vessel_dir, grade_path, grade_type)
    val_fundus_fns, val_vessel_fns, val_grades = load_fnames_grade(fundus_fns[:n_val], vessel_fns[:n_val], fundus_dir, vessel_dir, grade_path, grade_type)
    return (train_fundus_fns, train_vessel_fns, train_grades), (val_fundus_fns, val_vessel_fns, val_grades)


def split_dr(fundus_dirs, grade_path, val_ratio):
    fns = []
    for fundus_dir in fundus_dirs:
        fns += all_files_under(fundus_dir)

    random.Random(1).shuffle(fns)
    
    n_data = len(fns)
    n_val = int(n_data * val_ratio)

    train_fundus_fns, train_grades = load_fnames_grade_dr(fns[n_val:], grade_path)
    val_fundus_fns, val_grades = load_fnames_grade_dr(fns[:n_val], grade_path)
    return (train_fundus_fns, train_grades), (val_fundus_fns, val_grades)


def class_weight(grades):
    print "total data: {}".format(len(grades))
    grades = grades.astype(np.uint8)
    counts = np.bincount(grades)
    print "ratio: {}".format(counts)
    weight = np.array([1. / c for c in counts])
    weight/=sum(weight)
    print "class weight : {}".format(weight)
    return weight


def copy_file(src, dst):
    pipes = Popen(["cp", src, dst], stdout=PIPE, stderr=PIPE)
    std_out, std_err = pipes.communicate()
    if len(std_err) > 0:
        print std_err


def load_fnames_grade_dr(fundus_fns, grade_path):
    # get points
    df_grade = pd.read_csv(grade_path)
    label_dict = dict(zip(df_grade["fname"], df_grade["grade"]))
    new_fundus_filepaths = []
    grades = []
    for index in range(len(fundus_fns)):
        fname = fundus_fns[index]
        if fname in label_dict:
            grade = label_dict[fname]
            grades.append(grade)
            new_fundus_filepaths.append(fundus_fns[index])
 
    return np.array(new_fundus_filepaths), np.array(grades)

            
def load_fnames_grade(fundus_fns, vessel_fns, fundus_dir, vessel_dir, grade_path, grade_type):
    # get points
    df_grade = pd.read_csv(grade_path)
    if grade_type == "DR":
        label_dict = dict(zip(df_grade["Image name"], df_grade["Retinopathy grade"]))
    elif grade_type == "DME":
        label_dict = dict(zip(df_grade["Image name"], df_grade["Risk of macular edema "]))
    new_fundus_filepaths = []
    new_vessel_filepaths = []
    grades = []
    for index in range(len(fundus_fns)):
        fname = fundus_fns[index]
        if "IDRiD" in fname:
            fname = fname.replace(".tif", "")
        if fname in label_dict:
            grade = label_dict[fname]
            grades.append(grade)
            new_fundus_filepaths.append(os.path.join(fundus_dir, fundus_fns[index]))
            new_vessel_filepaths.append(os.path.join(vessel_dir, vessel_fns[index]))
 
    return np.array(new_fundus_filepaths), np.array(new_vessel_filepaths), np.array(grades)


def outputs2labels(outputs, min_val, max_val):
    return np.clip(np.round(outputs), min_val, max_val)


def best_threshold(true, pred, start_label):
    # binary between start_label and start_labe+1
    true=np.array(true)
    true_tmp = np.zeros((len(true),))
    true_tmp[true > start_label+0.1] = 1
    true_tmp[true <= start_label+0.1] = 0
    
    sorted_pred = sorted(pred)
    thresholds = np.unique(np.clip(sorted_pred, start_label, start_label + 1))
    best_acc = 0
    for th in thresholds:
        pred_tmp = np.zeros((len(pred),))
        pred_tmp[pred >= th] = 1
        pred_tmp[pred < th] = 0
        if len(pred_tmp[pred_tmp == 0]) == 0 or len(pred_tmp[pred_tmp == 1]) == 0:
            continue 
        cm = confusion_matrix(true_tmp, pred_tmp)
        n_total = np.sum(cm)
        correct = np.sum([cm[i, i] for i in range(2)])
        acc = 1.*correct / n_total
        if acc > best_acc:
            best_th = th
            best_acc = acc
    print best_th, best_acc
    return best_th

            
def adjust_threshold(true_grades, pred_grades):
    threshold_pred_grades = np.zeros((len(pred_grades),))
    pred_grades = np.array(pred_grades)
    
    best_ths = []
    for i in range(4):
        best_ths.append(best_threshold(true_grades, pred_grades, i))
    
    threshold_pred_grades[pred_grades < best_ths[0]] = 0
    for index in range(len(best_ths) - 1):
        threshold_pred_grades[(pred_grades >= best_ths[index]) & (pred_grades < best_ths[index + 1])] = index + 1
    threshold_pred_grades[(pred_grades >= best_ths[3]) ] = 4
    return threshold_pred_grades


def print_confusion_matrix(true_grades, pred_grades, grade_type):
    if grade_type == "DR":
        min_label, max_label = 0, 4
    elif grade_type == "DME":
        min_label, max_label = 0, 2
    n_label = max_label - min_label + 1
    cm = confusion_matrix(true_grades, outputs2labels(pred_grades, min_label, max_label))
    n_total = np.sum(cm)
    correct = np.sum([cm[i, i] for i in range(min_label, n_label + min_label)])
    acc = 1.*correct / n_total
    
    print cm
    print "acc: {}".format(acc)


def save_wrong_files(true, pred, filenames, segmented_dir_tempalte, ori_img_dir):
    for i in range(len(true)):
        if true[i] != pred[i]:
            out_dir = "../outputs/dme/true{}_wrong{}".format(true[i], pred[i])
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            pipes = Popen(["cp", os.path.join(segmented_dir_tempalte.format(int(true[i])), filenames[i] + ".tif_overlap.png"),
                           os.path.join(out_dir, filenames[i] + ".tif_overlap.png")],
                            stdout=PIPE, stderr=PIPE)
            std_out, std_err = pipes.communicate()
            pipes = Popen(["cp", os.path.join(segmented_dir_tempalte.format(int(true[i])), filenames[i] + ".tif_EX_segmented.png"),
                           os.path.join(out_dir, filenames[i] + ".tif_EX_segmented.png")],
                            stdout=PIPE, stderr=PIPE)
            pipes = Popen(["cp", os.path.join(ori_img_dir, filenames[i] + ".tif"),
                           os.path.join(out_dir, filenames[i] + ".tif")],
                            stdout=PIPE, stderr=PIPE)
            std_out, std_err = pipes.communicate()
            

def load_augmented_fundus_vessel(fundus_fnames, vessel_fnames, augment, normalize):
    assert len(fundus_fnames) == len(vessel_fnames)
    n_imgs = len(fundus_fnames)
    img_shape = image_shape(fundus_fnames[0]) 
    fundus_z_arr = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
    fundus_normalized = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
    vessel_arr = np.zeros((n_imgs, img_shape[0], img_shape[1], 1))
        
    for file_index in xrange(n_imgs):
        fundus = np.array(Image.open(fundus_fnames[file_index])).astype(np.float32) / 255.
        vessel = np.array(Image.open(vessel_fnames[file_index])).astype(np.float32) / 255.

        if augment:
            # random addition of optic disc like structure 
            # random flip
            if random.getrandbits(1):
                fundus = fundus[:, ::-1]
                vessel = vessel[:, ::-1]

            # affine transform (translation, scale, shear, rotation)
            r_angle = random.randint(0, 359)
            x_scale, y_scale = random.uniform(1. / 1.05, 1.05), random.uniform(1. / 1.05, 1.05)
            shear = random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
            x_translation, y_translation = random.randint(-10, 10), random.randint(-10, 10)
            tform = AffineTransform(scale=(x_scale, y_scale), shear=shear,
                                    translation=(x_translation, y_translation))
            fundus_warped = warp(fundus, tform.inverse, output_shape=(fundus.shape[0], fundus.shape[1]))
            vessel_warped = warp(vessel, tform.inverse, output_shape=(vessel.shape[0], vessel.shape[1]))
            fundus = rotate(fundus_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            vessel = rotate(vessel_warped, r_angle, axes=(0, 1), order=1, reshape=False)
        
        # convert to z score per image
        means, stds = np.zeros(3), np.array([1.0, 1.0, 1.0])
        for i in range(3):
            if len(fundus[..., i][fundus[..., i] > 10. / 255]) > 1:
                means[i] = np.mean(fundus[..., i][fundus[..., i] > 10. / 255])
                std_val = np.std(fundus[..., i][fundus[..., i] > 10. / 255])
                stds[i] = std_val if std_val > 0 else 1

        if normalize == "rescale_mean_subtract":
            fundus_normalized[file_index] = fundus - means
        elif normalize == "rescale":    
            fundus_normalized[file_index] = fundus
        
        fundus_z_arr[file_index] = (fundus - means) / stds
        vessel_arr[file_index, ...] = np.round(np.expand_dims(vessel, axis=2))
            
    return fundus_normalized, fundus_z_arr, vessel_arr


def load_augmented(fundus_fnames, augment):
    n_imgs = len(fundus_fnames)
    img_shape = image_shape(fundus_fnames[0]) 
    fundus_rescale = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
    fundus_rescale_mean_subtracted = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
        
    for file_index in xrange(n_imgs):
        fundus = np.array(Image.open(fundus_fnames[file_index])).astype(np.float32) / 255.

        if augment:
            # random addition of optic disc like structure 
            # random flip
            if random.getrandbits(1):
                fundus = fundus[:, ::-1]

            # affine transform (translation, scale, shear, rotation)
            r_angle = random.randint(0, 359)
            x_scale, y_scale = random.uniform(1. / 1.05, 1.05), random.uniform(1. / 1.05, 1.05)
            shear = random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
            x_translation, y_translation = random.randint(-10, 10), random.randint(-10, 10)
            tform = AffineTransform(scale=(x_scale, y_scale), shear=shear,
                                    translation=(x_translation, y_translation))
            fundus_warped = warp(fundus, tform.inverse, output_shape=(fundus.shape[0], fundus.shape[1]))
            fundus = rotate(fundus_warped, r_angle, axes=(0, 1), order=1, reshape=False)
        
        # convert to z score per image
        means = np.zeros(3)
        for i in range(3):
            if len(fundus[..., i][fundus[..., i] > 10. / 255]) > 1:
                means[i] = np.mean(fundus[..., i][fundus[..., i] > 10. / 255])

        fundus_rescale = fundus
        fundus_rescale_mean_subtracted[file_index] = fundus - means
            
    return fundus_rescale, fundus_rescale_mean_subtracted


def load_augmented_segmentation_as_input(fundus_fnames, segmentation_home, augment):
    n_imgs = len(fundus_fnames)
    img_shape = image_shape(fundus_fnames[0]) 
    fundus_rescale_mean_subtracted = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
    ex1 = np.zeros((n_imgs, img_shape[0], img_shape[1], 1))
    he1 = np.zeros((n_imgs, img_shape[0], img_shape[1], 1))
    ma1 = np.zeros((n_imgs, img_shape[0], img_shape[1], 1))
    se1 = np.zeros((n_imgs, img_shape[0], img_shape[1], 1))
    fundus_rescale_mean_subtracted_lesions = np.zeros((n_imgs, img_shape[0], img_shape[1], 7))
    
    for file_index in xrange(n_imgs):
        fundus = np.array(Image.open(fundus_fnames[file_index])).astype(np.float32) / 255.
        f_basename = os.path.basename(fundus_fnames[file_index])
        ex = np.array(Image.open(os.path.join(segmentation_home, "EX_512", f_basename))).astype(np.float32) / 255.
        he = np.array(Image.open(os.path.join(segmentation_home, "HE_512", f_basename))).astype(np.float32) / 255.
        ma = np.array(Image.open(os.path.join(segmentation_home, "MA_512", f_basename))).astype(np.float32) / 255.
        se = np.array(Image.open(os.path.join(segmentation_home, "SE_512", f_basename))).astype(np.float32) / 255.
        
        if augment:
            # random addition of optic disc like structure 
            # random flip
            if random.getrandbits(1):
                fundus = fundus[:, ::-1]
                ex = ex[:, ::-1]
                he = he[:, ::-1]
                ma = ma[:, ::-1]
                se = se[:, ::-1]

            # affine transform (translation, scale, shear, rotation)
            r_angle = random.randint(0, 359)
            x_scale, y_scale = random.uniform(1. / 1.05, 1.05), random.uniform(1. / 1.05, 1.05)
            shear = random.uniform(-5 * np.pi / 180, 5 * np.pi / 180)
            x_translation, y_translation = random.randint(-10, 10), random.randint(-10, 10)
            tform = AffineTransform(scale=(x_scale, y_scale), shear=shear,
                                    translation=(x_translation, y_translation))
            fundus_warped = warp(fundus, tform.inverse, output_shape=(fundus.shape[0], fundus.shape[1]))
            ex_warped = warp(ex, tform.inverse, output_shape=(ex.shape[0], ex.shape[1]))
            he_warped = warp(he, tform.inverse, output_shape=(he.shape[0], he.shape[1]))
            ma_warped = warp(ma, tform.inverse, output_shape=(ma.shape[0], ma.shape[1]))
            se_warped = warp(se, tform.inverse, output_shape=(se.shape[0], se.shape[1]))
            
            fundus = rotate(fundus_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            ex = rotate(ex_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            he = rotate(he_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            ma = rotate(ma_warped, r_angle, axes=(0, 1), order=1, reshape=False)
            se = rotate(se_warped, r_angle, axes=(0, 1), order=1, reshape=False)
        
        # convert to z score per image
        means = np.zeros(3)
        for i in range(3):
            if len(fundus[..., i][fundus[..., i] > 10. / 255]) > 1:
                means[i] = np.mean(fundus[..., i][fundus[..., i] > 10. / 255])

        fundus_rescale_mean_subtracted[file_index] = fundus - means
        ex1[file_index, ..., 0] = ex  
        he1[file_index, ..., 0] = he  
        ma1[file_index, ..., 0] = ma  
        se1[file_index, ..., 0] = se  
            
        fundus_rescale_mean_subtracted_lesions[file_index, ...] = np.concatenate([fundus_rescale_mean_subtracted, ex1, he1, ma1, se1], axis=3)
            
    return fundus_rescale_mean_subtracted_lesions


def load_features_fundus(fundus_fnames, feature_shape_ex_he, feature_shape_ma, feature_shape_se, features_home, augment):
    n_imgs = len(fundus_fnames)
    img_shape = image_shape(fundus_fnames[0]) 
    fundus_rescale_mean_subtracted = np.zeros((n_imgs, img_shape[0], img_shape[1], 3))
    ex = np.zeros((n_imgs,) + feature_shape_ex_he)
    he = np.zeros((n_imgs,) + feature_shape_ex_he)
    ma = np.zeros((n_imgs,) + feature_shape_ma)
    se = np.zeros((n_imgs,) + feature_shape_se)
    
    for file_index in xrange(n_imgs):
        fundus = np.array(Image.open(fundus_fnames[file_index])).astype(np.float32) / 255.

        if augment:
            fundus = fundus * (1 + random.random() * 0.1) if random.getrandbits(1) else fundus * (1 - random.random() * 0.1)
        
        # convert to z score per image
        means = np.zeros(3)
        for i in range(3):
            if len(fundus[..., i][fundus[..., i] > 10. / 255]) > 1:
                means[i] = np.mean(fundus[..., i][fundus[..., i] > 10. / 255])

        ex[file_index, ...] = np.load(os.path.join(features_home, "EX", os.path.basename(fundus_fnames[file_index]) + ".npy"))
        he[file_index, ...] = np.load(os.path.join(features_home, "HE", os.path.basename(fundus_fnames[file_index]) + ".npy"))
        ma[file_index, ...] = np.load(os.path.join(features_home, "MA", os.path.basename(fundus_fnames[file_index]) + ".npy"))
        se[file_index, ...] = np.load(os.path.join(features_home, "SE", os.path.basename(fundus_fnames[file_index]) + ".npy"))
        
        fundus_rescale_mean_subtracted[file_index] = fundus - means
            
    return ex, he, ma , se, fundus_rescale_mean_subtracted


def dist(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def in_circle(pt, c_pt, r):
    return dist(pt, c_pt) <= r


def save_imgs_with_pts_fovea(imgs, pts, out_dir, uid):
    imgs = (10 * imgs + 100).astype(np.uint8)
    for index in range(imgs.shape[0]):
        imgs[index, int(imgs.shape[1] * pts[index, 0]) - 3:int(imgs.shape[1] * pts[index, 0]) + 3,
                    int(imgs.shape[2] * pts[index, 1]) - 3:int(imgs.shape[2] * pts[index, 1]) + 3, :] = (0, 0, 255)
        Image.fromarray(imgs[index, :, :].astype(np.uint8)).save(os.path.join(out_dir, str(uid) + "_{}.png".format(index)))


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


def crop_img(img, center, target_size):
    h_center, w_center = center[0], center[1]
    half_h, half_w = target_size[0] // 2, target_size[1] // 2 

    h_start, h_end = max(0, h_center - half_h), min(img.shape[0], h_center + half_h)
    w_start, w_end = max(0, w_center - half_w), min(img.shape[1], w_center + half_w)
    cropped = img[h_start:h_end, w_start:w_end, :]
    padded = pad_img(cropped, target_size)
    return padded


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


def save_figs_for_checks(segmented, fovea_activations, true_grades, img_out_dir, fnames):
    assert len(segmented.shape) == 4 and len(fovea_activations.shape) == 4 
    for index in range(segmented.shape[0]):
        fname = os.path.basename(fnames[index])
        grade_str = str(true_grades[index])
        dir_name = os.path.join(img_out_dir, grade_str)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        EX_segmented = (segmented[index, ..., 0] * 255).astype(np.uint8)
        fv_act = fovea_activations[index, ..., 0]
        fovea_activation = imresize((fv_act * 255).astype(np.uint8), (process_len, process_len), 'bilinear')
        fovea_activation_normalized = imresize((((fv_act - np.min(fv_act)) / (np.max(fv_act) - np.min(fv_act))) * 255).astype(np.uint8), (process_len, process_len), 'bilinear')

        Image.fromarray(EX_segmented).save(os.path.join(dir_name, fname + "_EX_segmented.png"))
        overlap = np.zeros((process_len, process_len, 3)).astype(np.uint8)
        overlap_activation_normalized = np.zeros((process_len, process_len, 3)).astype(np.uint8)
        overlap[..., 0] = fovea_activation
        overlap[..., 1] = EX_segmented
        overlap_activation_normalized[..., 0] = fovea_activation_normalized
        overlap_activation_normalized[..., 1] = EX_segmented
        Image.fromarray(overlap).save(os.path.join(dir_name, fname + "_overlap.png"))
        Image.fromarray(overlap_activation_normalized).save(os.path.join(dir_name, fname + "_overlap_activation_normalized.png"))


def save_figs_for_region_seg_check(segmented, od_segmentation, fovea_loc, true_grades, img_out_dir, fnames):
    assert len(segmented.shape) == 4 and len(od_segmentation.shape) == 4 
    y, x = np.ogrid[:process_len, :process_len]
    
    for index in range(segmented.shape[0]):
        fname = os.path.basename(fnames[index])
        grade_str = str(true_grades[index])
        dir_name = os.path.join(img_out_dir, grade_str)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        EX_segmented = (segmented[index, ..., 0] * 255).astype(np.uint8)
        Image.fromarray(EX_segmented).save(os.path.join(dir_name, fname + "_EX_segmented.png"))
        
        od = od_segmentation[index, ..., 0]
        od_resized = imresize((od * 255).astype(np.uint8), (process_len, process_len), 'bilinear')
        label, n_labels = measure.label(od_resized > 225 // 2, return_num=True)
        majority_vote = -1 if n_labels == 0 else np.argmax(np.bincount(label.flatten())[1:]) + 1
        pred_row_inds, pred_col_inds = np.where(label == majority_vote)
        if len(pred_row_inds) > 0:
            radius = max(np.max(pred_row_inds) - np.min(pred_row_inds), np.max(pred_col_inds) - np.min(pred_col_inds)) / 2
        else:
            print "od not detected : {}".format(os.path.basename(fnames[index]))
            radius = 50
            
        fovea_center = (fovea_loc[index, 0] * process_len, fovea_loc[index, 1] * process_len)
        fovea_area = np.zeros((process_len, process_len))
        fovea_area[in_circle((y, x), fovea_center, radius)] = 255
        overlap = np.zeros((process_len, process_len, 3)).astype(np.uint8)
        overlap[..., 0] = fovea_area
        overlap[..., 2] = od_resized
        overlap[..., 1] = EX_segmented
        Image.fromarray(overlap).save(os.path.join(dir_name, fname + "_overlap.png"))


def extract_features(segmented, od_segmentation, fovea_loc):
    assert len(segmented.shape) == 4 and len(od_segmentation.shape) == 4 
    y, x = np.ogrid[:process_len, :process_len]
    
    features = [[] for _ in range(segmented.shape[0])]
        
    for index in range(segmented.shape[0]):
        od = od_segmentation[index, ..., 0]
        od_resized = imresize((od * 255).astype(np.uint8), (process_len, process_len), 'bilinear')
        label, n_labels = measure.label(od_resized > 225 // 2, return_num=True)
        majority_vote = -1 if n_labels == 0 else np.argmax(np.bincount(label.flatten())[1:]) + 1
        pred_row_inds, pred_col_inds = np.where(label == majority_vote)
        if len(pred_row_inds) > 0:
            radius = max(np.max(pred_row_inds) - np.min(pred_row_inds), np.max(pred_col_inds) - np.min(pred_col_inds)) / 2
            features[index].append(1)
        else:
            radius = 50
            features[index].append(0)
                
        fovea_center = (fovea_loc[index, 0] * process_len, fovea_loc[index, 1] * process_len)
        EX_segmented = segmented[index, ..., 0]
        out_dict = {}
        out_dict["ex_r"] = EX_segmented[in_circle((y, x), fovea_center, radius)]
        out_dict["ex_r_2r"] = EX_segmented[~(in_circle((y, x), fovea_center, radius)) & (in_circle((y, x), fovea_center, 2 * radius))]
        out_dict["ex_2r"] = EX_segmented[~in_circle((y, x), fovea_center, 2 * radius)]
        threshold = 178.0 / 255
        for key, value in sorted(out_dict.iteritems()):
            sum_int = np.sum(value) if len(value) > 0 else 0
            features[index].append(sum_int)  # sum

            value[value < threshold] = 0
            value_above_threshold = value[value > threshold]
            if len(value_above_threshold) == 0:
                label_maps = np.zeros(value.shape)
                n_labels = 0
            else:
                label_maps, n_labels = measure.label(value > threshold, return_num=True)
            label_maps = label_maps.astype(np.uint8)
            n_pixels = len(label_maps[label_maps > 0])  # pixel num
            majority_vote = -1 if n_labels == 0 else np.argmax(np.bincount(label_maps.flatten())[1:]) + 1
            minority_vote = -1 if n_labels == 0 else np.argmin(np.bincount(label_maps.flatten())[1:]) + 1
            largest = 0 if n_labels == 0 else len(label_maps[label_maps == majority_vote])  # pixel num of largest blob
            smallest = 0 if n_labels == 0 else len(label_maps[label_maps == minority_vote])  # pixel num of smallest blob
            mean_blob_size = 0 if n_labels == 0 else np.mean([len(label_maps[label_maps == l]) for l in range(1, n_labels + 1)])  # average pixel num of blobs
            features[index].append(n_pixels)
            features[index].append(largest)  
            features[index].append(smallest)
            features[index].append(mean_blob_size)
    
    return np.array(features)


def extract_features_dr(ex_arr, he_arr, ma_arr, se_arr):
    
    features = [[] for _ in range(ex_arr.shape[0])]
        
    for index in range(ex_arr.shape[0]):
        EX_segmented = ex_arr[index, ..., 0]
        HE_segmented = he_arr[index, ..., 0]
        MA_segmented = ma_arr[index, ..., 0]
        SE_segmented = se_arr[index, ..., 0]
        
        out_dict = {"EX":EX_segmented, "HE":HE_segmented, "MA":MA_segmented, "SE":SE_segmented}        
        thresholds = {"EX": 230.0 / 255, "HE": 246.0 / 255, "MA":242.0 / 255 , "SE":212.0 / 255}
        for key, value in sorted(out_dict.iteritems()):
            sum_int = np.sum(value) if len(value) > 0 else 0
            features[index].append(sum_int)  # sum

            value[value < thresholds[key]] = 0
            value_above_threshold = value[value > thresholds[key]]
            if len(value_above_threshold) == 0:
                label_maps = np.zeros(value.shape)
                n_labels = 0
            else:
                label_maps, n_labels = measure.label(value > thresholds[key], return_num=True)
            label_maps = label_maps.astype(np.uint8)
            n_pixels = len(label_maps[label_maps > 0])  # pixel num
            majority_vote = -1 if n_labels == 0 else np.argmax(np.bincount(label_maps.flatten())[1:]) + 1
            minority_vote = -1 if n_labels == 0 else np.argmin(np.bincount(label_maps.flatten())[1:]) + 1
            largest = 0 if n_labels == 0 else len(label_maps[label_maps == majority_vote])  # pixel num of largest blob
            smallest = 0 if n_labels == 0 else len(label_maps[label_maps == minority_vote])  # pixel num of smallest blob
            mean_blob_size = 0 if n_labels == 0 else np.mean([len(label_maps[label_maps == l]) for l in range(1, n_labels + 1)])  # average pixel num of blobs
            features[index].append(n_pixels)
            features[index].append(largest)  
            features[index].append(smallest)
            features[index].append(mean_blob_size)
    
    return np.array(features)

# def extract_features(segmented, od_segmentation, fovea_loc):
#     assert len(segmented.shape) == 4 and len(od_segmentation.shape) == 4 
#     y, x = np.ogrid[:process_len, :process_len]
#     
#     features = [[] for _ in range(segmented.shape[0])]
#         
#     for index in range(segmented.shape[0]):
#         od = od_segmentation[index, ..., 0]
#         od_resized = imresize((od * 255).astype(np.uint8), (process_len, process_len), 'bilinear')
#         label = measure.label(od_resized > 225 // 2, return_num=False)
#         pred_row_inds, pred_col_inds = np.where(label == 1)
#         if len(pred_row_inds) > 0:
#             radius = max(np.max(pred_row_inds) - np.min(pred_row_inds), np.max(pred_col_inds) - np.min(pred_col_inds)) / 2
#             features[index].append(1)
#         else:
#             radius = 50
#             features[index].append(0)
#                 
#         fovea_center = (fovea_loc[index, 0] * process_len, fovea_loc[index, 1] * process_len)
#         EX_segmented = segmented[index, ..., 0]
#         out_dict = {}
#         out_dict["ex"] = EX_segmented
#         out_dict["ex_r"] = EX_segmented[in_circle((y, x), fovea_center, radius)]
#         out_dict["ex_r_2r"] = EX_segmented[~(in_circle((y, x), fovea_center, radius)) & (in_circle((y, x), fovea_center, 2 * radius))]
#         out_dict["ex_2r"] = EX_segmented[~in_circle((y, x), fovea_center, 2 * radius)]
#         threshold = 178.0 / 255
#         for key, value in sorted(out_dict.iteritems()):
#             sum_int = np.sum(value) if len(value) > 0 else 0
#             mean_int = np.mean(value) if len(value) > 0 else 0
#             features[index].append(sum_int)  # sum
#             features[index].append(mean_int)  # mean
#             if key == "ex":
#                 entire_sum = sum_int
#                 entire_mean = mean_int
#             else:
#                 int_ratio = 1.*sum_int / entire_sum if entire_sum != 0 else 0
#                 mean_ratio = 1.*mean_int / entire_mean if entire_mean != 0 else 0
#                 features[index].append(int_ratio)
#                 features[index].append(mean_ratio)
#             
#             value[value < threshold] = 0
#             value_above_threshold = value[value > threshold]
#             if len(value_above_threshold) == 0:
#                 label_maps = np.zeros(value.shape)
#                 n_labels = 0
#             else:
#                 label_maps, n_labels = measure.label(value > threshold, return_num=True)
#             n_pixels = len(label_maps[label_maps > 0])  # pixel num
#             n_blobs = n_labels  # blob num
#             largest = len(label_maps[label_maps == 1])  # pixel num of largest blob
#             smallest = len(label_maps[label_maps == n_labels])  # pixel num of smallest blob
#             mean_blob_size = 0 if n_labels == 0 else np.mean([len(label_maps[label_maps == l]) for l in range(1, n_labels + 1)])  # average pixel num of blobs
#             features[index].append(n_pixels)
#             features[index].append(n_blobs)
#             features[index].append(largest)  
#             features[index].append(smallest)
#             features[index].append(mean_blob_size)
#             if key == "ex":
#                 entire_n_pixels = n_pixels
#                 entire_n_blobs = n_blobs
#                 entire_largest = largest
#                 entire_smallest = smallest
#                 entire_mean_blob_size = mean_blob_size
#             else:
#                 n_pixels_ratio = 1.*n_pixels / entire_n_pixels if entire_n_pixels != 0 else 0
#                 n_blobs_ratio = 1.* n_blobs / entire_n_blobs  if  entire_n_blobs != 0 else 0
#                 largest_ratio = 1.* largest / entire_largest  if  entire_largest != 0 else 0
#                 smallest_ratio = 1.* smallest / entire_smallest  if  entire_smallest != 0 else 0
#                 mean_blob_size_ratio = 1.* mean_blob_size / entire_mean_blob_size  if  entire_mean_blob_size != 0 else 0
#                 features[index].append(n_pixels_ratio)
#                 features[index].append(n_blobs_ratio)
#                 features[index].append(largest_ratio)
#                 features[index].append(smallest_ratio)
#                 features[index].append(mean_blob_size_ratio)
# 
#     return np.array(features)

    
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
