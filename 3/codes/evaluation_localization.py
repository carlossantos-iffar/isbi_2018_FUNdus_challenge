import numpy as np
import pandas as pd
import argparse

# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gt_od',
    type=str,
    required=False
    )
parser.add_argument(
    '--gt_fovea',
    type=str,
    required=False
    )
parser.add_argument(
    '--pred_od',
    type=str,
    required=False
    )
parser.add_argument(
    '--pred_fovea',
    type=str,
    required=False
    )
FLAGS, _ = parser.parse_known_args()

gt_od = pd.read_csv(FLAGS.gt_od).dropna(subset=["Image No"])
gt_fovea = pd.read_csv(FLAGS.gt_fovea).dropna(subset=["Image No"])
pred_od = pd.read_csv(FLAGS.pred_od)
pred_fovea = pd.read_csv(FLAGS.pred_fovea)

gt_od_image_no = gt_od.loc[:, "Image No"].tolist()
gt_fovea_image_no = gt_fovea.loc[:, "Image No"].tolist()
pred_od_image_no = pred_od.loc[:, "Image No"].tolist()
pred_fovea_image_no = pred_fovea.loc[:, "Image No"].tolist()
gt_od_pos = np.array(zip(gt_od.loc[:, "Y - Coordinate"].tolist(), gt_od.loc[:, "X- Coordinate"].tolist()))
gt_fovea_pos = np.array(zip(gt_fovea.loc[:, "Y - Coordinate"].tolist(), gt_fovea.loc[:, "X- Coordinate"].tolist()))
pred_od_pos = np.array(zip(pred_od.loc[:, "Y-Coordinate"].tolist(), pred_od.loc[:, "X-Coordinate"].tolist()))
pred_fovea_pos = np.array(zip(pred_fovea.loc[:, "Y-Coordinate"].tolist(), pred_fovea.loc[:, "X-Coordinate"].tolist()))

assert (np.array(gt_od_image_no) == np.array(pred_od_image_no)).all()
assert (np.array(gt_fovea_image_no) == np.array(pred_fovea_image_no)).all()

od_dist = np.linalg.norm(gt_od_pos - pred_od_pos, axis=1)
fovea_dist = np.linalg.norm(gt_fovea_pos - pred_fovea_pos, axis=1)

print "--- all dataset ---"
print "od mean Euclidean dist: {}".format(np.mean(od_dist))
print "fovea mean Euclidean dist: {}".format(np.mean(fovea_dist))

print "--- validation ---"
print "od mean Euclidean dist: {}".format(np.mean(od_dist[351:]))
print "fovea mean Euclidean dist: {}".format(np.mean(fovea_dist[351:]))

print "--- training ---"
print "od mean Euclidean dist: {}".format(np.mean(od_dist[:351]))
print "fovea mean Euclidean dist: {}".format(np.mean(fovea_dist[:351]))

for rank in range(5):
    print "{}th worst od: {}".format(rank, gt_od_image_no[np.argsort(od_dist)[-1 * rank - 1]])
for rank in range(5):
    print "{}th worst fovea: {}".format(rank, gt_fovea_image_no[np.argsort(fovea_dist)[-1 * rank - 1]])
