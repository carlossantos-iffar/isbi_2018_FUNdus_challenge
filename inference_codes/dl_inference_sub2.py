import numpy as np
import utils_sub2
import os
from keras import backend as K
import pandas as pd
from PIL import Image
import time
import xgboost as xgb
import pickle


def dme_feature_extractor(fundus_dir):
    # set misc paths
    EX_segmentor_dir = "model_sub2/EX_segmentor"
    fovea_localizer_dir = "model_sub2/fovea_localizer"
    od_segmentor_dir = "model_sub2/od_segmentor"
    vessel_model_dir = "model_sub2/vessel"
    
    # load networks
    K.set_learning_phase(False)
    EX_segmentor = utils_sub2.load_network(EX_segmentor_dir)
    fovea_localizer = utils_sub2.load_network(fovea_localizer_dir)
    od_segmentor = utils_sub2.load_network(od_segmentor_dir)
    vessel_model = utils_sub2.load_network(vessel_model_dir)
    
    # iterate for each file
    filenames = utils_sub2.all_files_under(fundus_dir)
    list_fnames, list_features = [], []
    for filename in filenames:
        start_time = time.time()
    
        # load imgs
        fundus = np.array(Image.open(filename)).astype(np.float32)
        img = np.expand_dims(fundus, axis=0)
        resized_img_bicubic = utils_sub2.resize_img(img, "bicubic")
        resized_img_bilinear = utils_sub2.resize_img(img, "bilinear")
    
        # run infererence
        vessel = vessel_model.predict(utils_sub2.normalize(resized_img_bilinear, "vessel_segmentation"), batch_size=1, verbose=0)
        segmented = EX_segmentor.predict(utils_sub2.normalize(resized_img_bicubic, "ex_segmentor"), batch_size=1, verbose=0)
        fovea_loc, _ = fovea_localizer.predict([utils_sub2.normalize(resized_img_bilinear, "od_fv_localization"), (vessel * 255).astype(np.uint8) / 255.], batch_size=1, verbose=0)
        od_seg, _ = od_segmentor.predict([utils_sub2.normalize(resized_img_bilinear, "od_segmentation"), (vessel * 255).astype(np.uint8) / 255.], batch_size=1, verbose=0)

        # extract featuers
        features = utils_sub2.extract_features(segmented, od_seg, fovea_loc)
        list_fnames.append(os.path.basename(filename).replace(".jpg", ""))
        list_features.append(features)
        utils_sub2.save_figs_for_region_seg_check(segmented, od_seg, fovea_loc, "img_check_sub2", [filename])
        
        print "duration: {} sec".format(time.time() - start_time)
    
    features_matrix = np.concatenate(list_features, axis=0)
    out_dict = {}
    for index in range(features_matrix.shape[1]):
        out_dict[str(index)] = features_matrix[:, index]
    out_dict['fname'] = list_fnames
    df = pd.DataFrame(out_dict)
    utils_sub2.make_new_dir("outputs_sub2")
    df.to_csv("outputs_sub2/dme_features.csv", index=False)
    
    
def dme_feature_extractor_for_xgb_training(fundus_dir):
    # set misc paths
    EX_segmentor_dir = "model_sub2/EX_segmentor"
    fovea_localizer_dir = "model_sub2/fovea_localizer"
    od_segmentor_dir = "model_sub2/od_segmentor"
    vessel_model_dir = "model_sub2/vessel"
    grade_path = "xgb_training_sub2/IDRiD_Training Set.csv"

    # read labels
    df_grade = pd.read_csv(grade_path)
    label_dict = dict(zip(df_grade["Image name"], df_grade["Risk of macular edema "]))
    
    # load networks
    K.set_learning_phase(False)
    EX_segmentor = utils_sub2.load_network(EX_segmentor_dir)
    fovea_localizer = utils_sub2.load_network(fovea_localizer_dir)
    od_segmentor = utils_sub2.load_network(od_segmentor_dir)
    vessel_model = utils_sub2.load_network(vessel_model_dir)
    
    # iterate for each file
    filenames = utils_sub2.all_files_under(fundus_dir)
    list_fnames, list_features, list_grades = [], [], []
    for filename in filenames:
        start_time = time.time()
    
        # load imgs
        fundus = np.array(Image.open(filename)).astype(np.float32)
        img = np.expand_dims(fundus, axis=0)
        resized_img_bicubic = utils_sub2.resize_img(img, "bicubic")
        resized_img_bilinear = utils_sub2.resize_img(img, "bilinear")
    
        # run infererence
        vessel = vessel_model.predict(utils_sub2.normalize(resized_img_bilinear, "vessel_segmentation"), batch_size=1, verbose=0)
        segmented = EX_segmentor.predict(utils_sub2.normalize(resized_img_bicubic, "ex_segmentor"), batch_size=1, verbose=0)
        fovea_loc, _ = fovea_localizer.predict([utils_sub2.normalize(resized_img_bilinear, "od_fv_localization"), (vessel * 255).astype(np.uint8) / 255.], batch_size=1, verbose=0)
        od_seg, _ = od_segmentor.predict([utils_sub2.normalize(resized_img_bilinear, "od_segmentation"), (vessel * 255).astype(np.uint8) / 255.], batch_size=1, verbose=0)

        # extract featuers
        features = utils_sub2.extract_features(segmented, od_seg, fovea_loc)
        list_fnames.append(os.path.basename(filename).replace(".jpg", ""))
        list_features.append(features)
        grade = label_dict[os.path.basename(filename).replace(".jpg", "")]
        list_grades.append(grade)
        utils_sub2.save_figs_for_region_seg_check(segmented, od_seg, fovea_loc, "img_check_sub2", [filename])
        
        print "duration: {} sec".format(time.time() - start_time)
    
    features_matrix = np.concatenate(list_features, axis=0)
    out_dict = {}
    for index in range(features_matrix.shape[1]):
        out_dict[str(index)] = features_matrix[:, index]
    out_dict['fname'] = list_fnames
    out_dict['grade'] = list_grades
    df = pd.DataFrame(out_dict)
    utils_sub2.make_new_dir("outputs_sub2")
    df.to_csv("xgb_training_sub2/dme_features.csv", index=False)

    
def dr_inference(fundus_dir):
    # set misc paths
    dr_model_dir = "model_sub2/dr"
    
    # load networks
    K.set_learning_phase(False)
    dr_model = utils_sub2.load_network(dr_model_dir)
    
    # iterate for each file
    filenames = utils_sub2.all_files_under(fundus_dir)
    list_fnames, list_grades = [], []
    for filename in filenames:
        start_time = time.time()
    
        # load imgs
        fundus = np.array(Image.open(filename)).astype(np.float32)
        img = np.expand_dims(fundus, axis=0)
        resized_img_bicubic = utils_sub2.resize_img_dr(img, "bicubic")
    
        # run infererence
        dr_grade = dr_model.predict(utils_sub2.normalize(resized_img_bicubic, "dr"), batch_size=1, verbose=0)
        list_grades.append(int(utils_sub2.outputs2labels(dr_grade[0,0], 0, 4)))
        list_fnames.append(os.path.basename(filename).replace(".jpg", ""))
        print "duration: {} sec".format(time.time() - start_time)
    
    # output results
    df = pd.DataFrame({"Image No":list_fnames, "DR Grade":list_grades})
    df.to_csv("outputs_sub2/dr.csv", index=False)
    
def dme_xgb():
    # load features, labels
    csv_file = "outputs_sub2/dme_features.csv"
    df = pd.read_csv(csv_file)
    df_mat = df.as_matrix()
    
    # load models
    model_dir = "model_sub2/dme_xgb_models"
    models = []
    for i in range(10):
        model_path = os.path.join(model_dir, "model_{}".format(i))
        models.append(pickle.load(open(model_path)))
    
    # set dataset
    all_X = df_mat[:, :-1].astype(np.float32)
    X = xgb.DMatrix(all_X)
    
    # predict with the models
    all_preds = []
    for i in range(10):
        preds = models[i].predict(X)
        all_preds.append(np.expand_dims(preds, axis=1))
    answers = np.concatenate(all_preds, axis=1).astype(np.uint8)
    
    # majority vote
    final_answer = [np.argmax(np.bincount(answers[i, :])) for i in range(answers.shape[0])]
    
    # output results
    df = pd.DataFrame({"Image No":df_mat[:, -1], "Risk of DME":final_answer})
    df.to_csv("outputs_sub2/dme.csv", index=False)