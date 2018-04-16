import pandas as pd
import os

df_kaggle_train = pd.read_csv("../data/trainLabels.csv")
df_kaggle_test = pd.read_csv("../data/testLabels.csv")
df_given_data = pd.read_csv("../data/IDRiD_Training Set.csv")

out_csv = "../data/all_labels.csv"

df_kaggle_train.loc[:, "fname"] = df_kaggle_train.image.map(lambda x:os.path.join("../data/kaggle_DR_train/preprocessed/", x + ".png"))
df_kaggle_train.loc[:, "grade"] = df_kaggle_train.level.map(lambda x:x)
df_kaggle_test.loc[:, "fname"] = df_kaggle_test.image.map(lambda x:os.path.join("../data/kaggle_DR_test/preprocessed/", x + ".png"))
df_kaggle_test.loc[:, "grade"] = df_kaggle_test.level.map(lambda x:x)
df_given_data.loc[:, "fname"] = df_given_data["Image name"].map(lambda x:os.path.join("../data/Training_Set_preprocessed/", x + ".tif"))
df_given_data.loc[:, "grade"] = df_given_data["Retinopathy grade"].map(lambda x:x)

df_final = pd.concat([df_kaggle_train[["fname","grade"]], df_kaggle_test[["fname","grade"]], 
                      df_given_data[["fname","grade"]]], ignore_index=True)
df_final.to_csv(out_csv, index=False)
