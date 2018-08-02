import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import pickle
import utils

# load features, labels
csv_file = "../outputs/dme_features.csv"
df = pd.read_csv(csv_file)
df_mat = df.as_matrix()
n_total = len(df)

# load models
model_dir = "../dme_xgb_models"
models = []
for i in range(10):
    model_path = os.path.join(model_dir, "model_{}".format(i))
    models.append(pickle.load(open(model_path)))

# set dataset
all_X = df_mat[:, :-2].astype(np.float32)
all_Y = df_mat[:, -1].astype(np.float32)
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
df = pd.DataFrame({"Image No":df_mat[-413:, -2], "Risk of DME":final_answer[-413:]})
df.to_csv("VRT_Disease Grading.csv", index=False)
segmented_dir_tempalte = "../outputs/check_segmentation_fovea/{}/"
ori_img_dir = "../data/merged_training_set/"
utils.save_wrong_files(all_Y, final_answer, df_mat[:, -2], segmented_dir_tempalte, ori_img_dir)
     
# compute accuracy    
cm = confusion_matrix(all_Y, final_answer)
acc = 1.*(cm[0, 0] + cm[1, 1] + cm[2, 2]) / np.sum(cm)
print cm
print "accuracy : {}".format(acc)
