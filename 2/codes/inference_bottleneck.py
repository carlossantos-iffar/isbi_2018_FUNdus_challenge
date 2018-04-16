import argparse
import os

import iterator_dr
import model
import numpy as np
import utils
import multiprocessing


def process(args):
    valset, gpu_index, batch_size = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

    # set iterators for training and validation
    val_batch_fetcher = iterator_dr.ValidationBatchFetcher(valset, batch_size)
    
    # load networks
    EX_segmentor_dir = "../model/EX_segmentor"
    HE_segmentor_dir = "../model/HE_segmentor"
    MA_segmentor_dir = "../model/MA_segmentor"
    SE_segmentor_dir = "../model/SE_segmentor"
    EX_segmentor = utils.load_network(EX_segmentor_dir)
    HE_segmentor = utils.load_network(HE_segmentor_dir)
    MA_segmentor = utils.load_network(MA_segmentor_dir)
    SE_segmentor = utils.load_network(SE_segmentor_dir)
    EX_bottleneck_extractor = model.bottleneck_extractor(EX_segmentor, "activation_14", 0)
    HE_bottleneck_extractor = model.bottleneck_extractor(HE_segmentor, "activation_14", 0)
    MA_bottleneck_extractor = model.bottleneck_extractor(MA_segmentor, "activation_6", 4)
    SE_bottleneck_extractor = model.bottleneck_extractor(SE_segmentor, "activation_10", 2)

    for fnames, fundus_rescale, fundus_rescale_mean_subtract, _ in val_batch_fetcher():
        ex_bottleneck = EX_bottleneck_extractor.predict(fundus_rescale, batch_size=batch_size, verbose=0)
        he_bottleneck = HE_bottleneck_extractor.predict(fundus_rescale_mean_subtract, batch_size=batch_size, verbose=0)
        ma_bottleneck = MA_bottleneck_extractor.predict(fundus_rescale_mean_subtract, batch_size=batch_size, verbose=0)
        se_bottleneck = SE_bottleneck_extractor.predict(fundus_rescale, batch_size=batch_size, verbose=0) 
        
        for index in range(ex_bottleneck.shape[0]):
            np.save(os.path.join(out_dirs["EX"], os.path.basename(fnames[index])), ex_bottleneck[index, ...])
            np.save(os.path.join(out_dirs["HE"], os.path.basename(fnames[index])), he_bottleneck[index, ...])
            np.save(os.path.join(out_dirs["MA"], os.path.basename(fnames[index])), ma_bottleneck[index, ...])
            np.save(os.path.join(out_dirs["SE"], os.path.basename(fnames[index])), se_bottleneck[index, ...])


# arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size',
    type=int,
    help="batch size",
    required=True
    )
parser.add_argument(
    '--gpu_index',
    type=int,
    required=True
    )
parser.add_argument(
    '--available_gpus',
    type=int,
    required=True
    )
FLAGS, _ = parser.parse_known_args()

# set misc paths
fundus_dirs = ["../data/kaggle_DR_train/preprocessed", "../data/kaggle_DR_test/preprocessed", "../data/Training_Set_preprocessed"]
grade_path = "../data/all_labels.csv"
out_dir_template = "../data/bottleneck_features/{}"
out_dirs = {"EX":out_dir_template.format("EX"), "HE":out_dir_template.format("HE"),
          "MA":out_dir_template.format("MA"), "SE":out_dir_template.format("SE")}

for key, val in out_dirs.iteritems():
    if not os.path.isdir(val):
        os.makedirs(val)

_, validation_set = utils.split_dr(fundus_dirs, grade_path, 1)

# run multi-process
valsets = [[] for _ in xrange(FLAGS.available_gpus)]
chunk_sizes = len(validation_set[0]) // FLAGS.available_gpus
for index in xrange(FLAGS.available_gpus):
    if index == FLAGS.available_gpus - 1:  # allocate ranges (last GPU takes remainders)
        start, end = index * chunk_sizes, len(validation_set[0])
    else:
        start, end = index * chunk_sizes, (index + 1) * chunk_sizes
    valsets[index] = (validation_set[0][start:end], validation_set[1][start:end])

process((valsets[FLAGS.gpu_index], FLAGS.gpu_index, FLAGS.batch_size))

