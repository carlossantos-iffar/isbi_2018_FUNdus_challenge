import dl_inference_sub2
import argparse
import os
import utils_sub2

# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--fundus_dir',
    type=str,
    help="gpu index",
    required=True
    )
FLAGS, _ = parser.parse_known_args()

# training settings 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# dme
# dl_inference_sub2.dme_feature_extractor_for_xgb_training(FLAGS.fundus_dir)
dl_inference_sub2.dme_feature_extractor(FLAGS.fundus_dir)
dl_inference_sub2.dme_xgb()

# dr
dl_inference_sub2.dr_inference(FLAGS.fundus_dir)

# concat
utils_sub2.merge_csv_files()