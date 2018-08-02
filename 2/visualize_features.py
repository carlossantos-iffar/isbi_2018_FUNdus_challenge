import numpy as np
from PIL import Image
import argparse
import os

# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--filename',
    type=str,
    required=True,
    )
parser.add_argument(
    '--out_dir',
    type=str,
    required=True,
    )
FLAGS, _ = parser.parse_known_args()

arr = np.load(FLAGS.filename)

if not os.path.isdir(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

for i in range(arr.shape[2]):
    Image.fromarray((30*arr[..., i]).astype(np.uint8)).save(os.path.join(FLAGS.out_dir, "{}_{}.png".format(os.path.basename(FLAGS.filename),i)))
