# arrange arguments
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

csv_file = "../outputs/dme_features.csv"

# set font style
font = {'family':'serif'}
matplotlib.rc('font', **font)

# sort the order of plots manually for eye-pleasing plots
pt_type = ['r.', 'g.', 'b.']

# plot individual operation points
df = pd.read_csv(csv_file)
# df.loc[:,"IDRiD"]=df.fname.map(lambda x: "IDRiD" in x)
# df=df[~df["IDRiD"]]
for grade in range(3):
    sum_intensity_inside = df.loc[df["grade"]==grade, "sum_intensity_inside"]
    sum_intensity_outside = df.loc[df["grade"]==grade, "sum_intensity_outside"]
    plt.plot(np.log(sum_intensity_inside), np.log(sum_intensity_outside), pt_type[grade], label=grade)

plt.title('DME_grading')
plt.xlabel("sum_intensity_inside")
plt.ylabel("sum_intensity_outside")
# plt.xlim(0, 10)
# plt.ylim(0, 1000)
plt.legend(loc="lower right")
plt.savefig(os.path.join("../outputs", "dme_feature_draw.png"))
plt.close()
