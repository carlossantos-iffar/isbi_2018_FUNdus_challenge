import multiprocessing
import threading
import Queue
from uuid import uuid4

import numpy as np
import SharedArray

import utils
import random
    

def balance_per_class_indices(y):
    p = np.zeros(len(y))
    foreground = len(y[y == 1])
    background = len(y[y == 0])
    p[y == 0] = 1. / background
    p[y == 1] = 1. / foreground
    return np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                            p=np.array(p) / p.sum())


class BatchFetcher(object):

    def __init__(self, fundus_mask_pairs, batch_size, patch_size, augment, normalize):
        self.fundus, self.seg, self.labels = fundus_mask_pairs
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.augment = augment
        self.normalize=normalize
        super(BatchFetcher, self).__init__()
        
    def __call__(self):
        self.indices = np.array(range(self.labels.shape[0]))
        if self.augment:
            self.indices = balance_per_class_indices(self.labels)
        return self
  
    def __iter__(self):
        n_imgs = self.fundus.shape[0]
        bs = self.batch_size
        for i in range((n_imgs + bs - 1) // bs):
            sl = self.indices[i * bs:(i + 1) * bs]
            fundus_batch = self.fundus[sl, ...]
            seg_batch = self.seg[sl, ...]
            yield utils.load_augmented_patches(fundus_batch, seg_batch, self.patch_size, self.augment, self.normalize)
