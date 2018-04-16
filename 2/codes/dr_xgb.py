import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import time
import sys

# load features, labels
csv_file = "../outputs/dr_features.csv"
df = pd.read_csv(csv_file)
df_mat = df.as_matrix()
n_total = len(df)
ratio_val = 0.1
n_val = int(n_total * ratio_val)
train_X = df_mat[:-n_val, :-2].astype(np.float32)
train_Y = df_mat[:-n_val, -1].astype(np.float32)
val_X = df_mat[-n_val:, :-2].astype(np.float32)
val_Y = df_mat[-n_val:, -1].astype(np.float32)

# run xgboost
best_val_acc = 0
for subsample in [0.2, 0.4, 0.6]:
    for colsample_by_tree in [0.2, 0.4, 0.6]:
        for colsample_bylevel in [0.2, 0.4, 0.6]:
            for min_child_weight in [0.01, 0.1, 1, 2]:
                for lambda_val in range(1, 10, 2):
                    for alpha in range(1, 10, 2):
                        for depth in range(3, 10):
                            dtrain = xgb.DMatrix(train_X, label=train_Y)
                            dval = xgb.DMatrix(val_X)
                            param = {'max_depth':depth, 'subsample':subsample, 'colsample_by_tree': colsample_by_tree,
                                     "colsample_bylevel":colsample_bylevel, 'lambda':lambda_val, 'eta':0.1, 'alpha':alpha,
                                     'tree_method':"exact", 'num_class':5, 'min_child_weight':min_child_weight,
                                     'silent':1, 'objective':'multi:softmax' }
                            num_round = 100
                            bst = xgb.train(param, dtrain, num_round)
                            
                            # make prediction
                            preds = bst.predict(dval)
                            pred_labels = np.clip(np.round(preds), 0, 2)
                            cm = confusion_matrix(val_Y, pred_labels)
                            correct = np.sum([cm[i, i] for i in range(0, 4)])
                            acc = 1.*correct / np.sum(cm)
                            
                            if best_val_acc < acc:
                                best_val_acc = acc
                                print "min_child_weight:{}, subsample:{}, colsample_by_tree:{}, colsample_bylevel:{}, lambda_val:{}, alpha:{}, depth:{}".format(min_child_weight, subsample, colsample_by_tree, colsample_bylevel, lambda_val, alpha, depth)
                                print cm
                                print "validation accuracy : {}".format(acc)
                            
                                # check training phase
                                pred_labels = bst.predict(dtrain)
                                cm = confusion_matrix(train_Y, pred_labels)
                                correct = np.sum([cm[i, i] for i in range(0, 4)])
                                acc = 1.*correct / np.sum(cm)
                                print cm
                                print "training accuracy : {}".format(acc)
                                sys.stdout.flush()
                                                
