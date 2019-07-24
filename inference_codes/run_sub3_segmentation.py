import dl_inference_sub3
import argparse
import os

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
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# segmentation
dl_inference_sub3.segmentation_inference(FLAGS.fundus_dir)

