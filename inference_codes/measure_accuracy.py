import pandas as pd
import utils_sub2
    
df_dr_pred = pd.read_csv("outputs_sub2/dr.csv")
df_dme_pred = pd.read_csv("outputs_sub2/dme.csv")

df_true = pd.read_csv("xgb_training_sub2/IDRiD_Training Set.csv")

true_dr=df_true["Retinopathy grade"].tolist()
pred_dr=df_dr_pred["DR Grade"].tolist()

true_dme=df_true["Risk of macular edema "].tolist()
pred_dme=df_dme_pred["Risk of DME"].tolist()

utils_sub2.print_confusion_matrix(true_dr, pred_dr, "DR")
utils_sub2.print_confusion_matrix(true_dme, pred_dme, "DME")