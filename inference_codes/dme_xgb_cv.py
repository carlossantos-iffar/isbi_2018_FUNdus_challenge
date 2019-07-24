import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
from subprocess import Popen, PIPE
import operator
import pickle
import utils_sub2

# load features, labels
csv_file = "xgb_training_sub2/dme_features.csv"
df = pd.read_csv(csv_file)
df_mat = df.as_matrix()
n_total = len(df)
ratio_val = 0.1
n_val = int(n_total * ratio_val)

# set outdir
out_dir = "model_sub2/dme_xgb_models"
utils_sub2.make_new_dir(out_dir)

# run xgboost
min_child_weight = 0.01
subsample = 0.2
colsample_by_tree = 0.2
colsample_bylevel = 0.6
lambda_val = 1
alpha = 1
depth = 8

train_accs, val_accs = [], []

for i in range(10):
    # set training and validation dataset
    train_X = np.concatenate([df_mat[(i + 1) * n_val:, :-2].astype(np.float32), df_mat[:i * n_val, :-2].astype(np.float32)], axis=0)
    train_Y = np.concatenate([df_mat[(i + 1) * n_val:, -1].astype(np.float32), df_mat[:i * n_val, -1].astype(np.float32)], axis=0)
    val_X = df_mat[i * n_val:(i + 1) * n_val, :-2].astype(np.float32)
    val_Y = df_mat[i * n_val:(i + 1) * n_val, -1].astype(np.float32)
    
    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dval = xgb.DMatrix(val_X)
    param = {'max_depth':depth, 'subsample':subsample, 'colsample_by_tree': colsample_by_tree,
             "colsample_bylevel":colsample_bylevel, 'lambda':lambda_val, 'eta':0.1, 'alpha':alpha,
             'tree_method':"exact", 'num_class':3, 'min_child_weight':min_child_weight,
             'silent':1, 'objective':'multi:softmax' }
    num_round = 100
    bst = xgb.train(param, dtrain, num_round)
    pickle.dump(bst, open(os.path.join(out_dir, "model_{}".format(i)), "wb"))

    importance = bst.get_fscore()
    sum_importance = reduce(lambda x, y:x + y, importance.values()) 
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    print [(imp[0], 1.*imp[1] / sum_importance) for imp in importance]
    
    # make prediction
    preds = bst.predict(dval)
    pred_labels = np.clip(np.round(preds), 0, 2)
    cm = confusion_matrix(val_Y, pred_labels)
    val_acc = 1.*(cm[0, 0] + cm[1, 1] + cm[2, 2]) / np.sum(cm)
    print cm
    print "validation accuracy : {}".format(val_acc)
        
    # check training phase
    pred_labels = bst.predict(dtrain)
    cm = confusion_matrix(train_Y, pred_labels)
    train_acc = 1.*(cm[0, 0] + cm[1, 1] + cm[2, 2]) / np.sum(cm)
    print cm
    print "training accuracy : {}".format(train_acc)
    
    train_accs.append(train_acc)
    val_accs.append(val_acc)

print "mean train acc: {}".format(np.mean(train_accs))
print "std train acc: {}".format(np.std(train_accs, ddof=1))
print "mean val acc: {}".format(np.mean(val_accs))
print "std val acc: {}".format(np.std(val_accs, ddof=1))
