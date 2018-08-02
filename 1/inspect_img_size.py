from PIL import Image
from scipy.misc import imresize
import numpy as np

img_10_template = "../../data/original/{}/IDRiD_10.jpg"
img_19_template = "../../data/original/{}/IDRiD_19.jpg"

fundus_10 = np.array(Image.open(img_10_template.format("Apparent_Retinopathy")))
fundus_19 = np.array(Image.open(img_19_template.format("Apparent_Retinopathy")))

h, w, d = fundus_10.shape

img_10_template = "../../data/original/{}/IDRiD_10_{}.tif"
img_19_template = "../../data/original/{}/IDRiD_19_{}.tif"

HE_10 = np.array(Image.open(img_10_template.format("HE", "HE"))) * 255
HE_19 = np.array(Image.open(img_19_template.format("HE", "HE"))) * 255
    
SE_19 = np.array(Image.open(img_19_template.format("SE", "SE"))) * 255

MA_10 = np.array(Image.open(img_10_template.format("MA", "MA"))) * 255
MA_19 = np.array(Image.open(img_19_template.format("MA", "MA"))) * 255

EX_10 = np.array(Image.open(img_10_template.format("EX", "EX"))) * 255
EX_19 = np.array(Image.open(img_19_template.format("EX", "EX"))) * 255

for rate in [1. / 2, 1. / 4, 1. / 8, 1. / 16]:
    Image.fromarray(imresize(fundus_10, (int(h * rate), int(w * rate)), 'bilinear')).save("ex1_fundus_{}.png".format(str(rate)))
    Image.fromarray(imresize(fundus_19, (int(h * rate), int(w * rate)), 'bilinear')).save("ex2_fundus_{}.png".format(str(rate)))
    Image.fromarray(imresize(HE_10, (int(h * rate), int(w * rate)), 'bilinear')).save("HE_10_fundus_{}.png".format(str(rate)))
    Image.fromarray(imresize(HE_19, (int(h * rate), int(w * rate)), 'bilinear')).save("HE_19_fundus_{}.png".format(str(rate)))
    Image.fromarray(imresize(SE_19, (int(h * rate), int(w * rate)), 'bilinear')).save("SE_19_fundus_{}.png".format(str(rate)))
    Image.fromarray(imresize(MA_10, (int(h * rate), int(w * rate)), 'bilinear')).save("MA_10_fundus_{}.png".format(str(rate)))
    Image.fromarray(imresize(MA_19, (int(h * rate), int(w * rate)), 'bilinear')).save("MA_19_fundus_{}.png".format(str(rate)))
    Image.fromarray(imresize(EX_10, (int(h * rate), int(w * rate)), 'bilinear')).save("EX_10_fundus_{}.png".format(str(rate)))
    Image.fromarray(imresize(EX_19, (int(h * rate), int(w * rate)), 'bilinear')).save("EX_19_fundus_{}.png".format(str(rate)))
