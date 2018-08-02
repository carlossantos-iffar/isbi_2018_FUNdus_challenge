import pandas as pd
import utils
import os

# fundus_dir = "../data/Training_Set"
# grade_path = "../data/merged_labels.csv"
# out_dir = "../data/no_DR"

fundus_dir = "../data/merged_training_set"
grade_path = "../data/merged_labels.csv"
out_dir = "../data/merged_no_dr"
df_grade = pd.read_csv(grade_path)

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    
# df_grade_no_dr = df_grade[df_grade["Retinopathy grade"] == 0]
# for index, row in df_grade_no_dr.iterrows():
#     fname = row["Image name"]
#     if "IDRiD" in fname:
#         fname = fname + ".jpg"
#         utils.copy_file(os.path.join(fundus_dir, fname), os.path.join(out_dir, fname))

df_grade_no_dr = df_grade[df_grade["Retinopathy grade"] == 0]
for index, row in df_grade_no_dr.iterrows():
    fname = row["Image name"]
    if "IDRiD" in fname:
        fname = fname + ".tif"
    utils.copy_file(os.path.join(fundus_dir, fname), os.path.join(out_dir, fname))
