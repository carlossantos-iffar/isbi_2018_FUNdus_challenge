import multiprocessing
import threading
import Queue
from uuid import uuid4

import numpy as np
import SharedArray

import utils
import random

img_h, img_w = 640, 640


def load_shared(args):
    i, fundus_array_name, vessel_array_name, seg_array_name, augment, fundus_fname, vessel_fname, seg_fname = args
    fundus_array = SharedArray.attach(fundus_array_name)
    vessel_array = SharedArray.attach(vessel_array_name)
    seg_array = SharedArray.attach(seg_array_name)
    fundus_array[i], vessel_array[i], seg_array[i] = utils.load_augmented_fundus_vessel([fundus_fname], [vessel_fname], [seg_fname], augment)


def balance_per_class_indices(y, weights):
    weights = np.array(weights, dtype=float)
    p = np.zeros(len(y))
    for i, weight in enumerate(weights):
        p[y == i] = weight
    return np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                            p=np.array(p) / p.sum())


class BatchIterator(object):

    def __call__(self, *args):
        self.fundus_fnames, self.vessel_fnames, self.seg_fnames = args
        return self
        
    def __iter__(self):
        n_samples = self.fundus_fnames.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs - 1):
            sl = slice(i * bs, (i + 1) * bs)
            fundus_fnames = self.fundus_fnames[sl]
            vessel_fnames = self.vessel_fnames[sl]
            seg_fnames = self.seg_fnames[sl]
            yield self.transform(fundus_fnames, vessel_fnames, seg_fnames)


class QueueIterator(BatchIterator):

    def __iter__(self):
        queue = Queue.Queue(maxsize=20)
        end_marker = object()

        def producer():
            for filenames, funduses, vessels, segs in super(QueueIterator, self).__iter__():
                queue.put((np.array(filenames), np.array(funduses), np.array(vessels), np.array(segs)))
            queue.put(end_marker)
        
        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        item = queue.get()
        while item is not end_marker:
            yield item
            queue.task_done()
            item = queue.get()


class SharedIterator(QueueIterator):

    def __init__(self):
        self.pool = multiprocessing.Pool()

    def transform(self, fundus_fnames, vessel_fnames, seg_fnames):
        assert len(fundus_fnames) == len(vessel_fnames) and len(fundus_fnames) == len(seg_fnames) 
        n_imgs = len(fundus_fnames)
        fundus_shared_array_name = str(uuid4())
        vessel_shared_array_name = str(uuid4())
        seg_shared_array_name = str(uuid4())
        try:
            fundus_shared_array = SharedArray.create(
                fundus_shared_array_name, [n_imgs, img_h, img_w, 3], dtype=np.float32)
            vessel_shared_array = SharedArray.create(
                vessel_shared_array_name, [n_imgs, img_h, img_w, 1], dtype=np.float32)
            seg_shared_array = SharedArray.create(
                seg_shared_array_name, [len(seg_fnames), img_h, img_w, 1], dtype=np.float32)
                                        
            args = []
            
            for i in range(n_imgs):
                args.append((i, fundus_shared_array_name, vessel_shared_array_name, seg_shared_array_name,
                             self.augment, fundus_fnames[i], vessel_fnames[i], seg_fnames[i]))

            self.pool.map(load_shared, args)
            funduses = np.array(fundus_shared_array, dtype=np.float32)
            vessels = np.array(vessel_shared_array, dtype=np.float32)
            segs = np.array(seg_shared_array, dtype=np.float32)
            
        finally:
            SharedArray.delete(fundus_shared_array_name)
            SharedArray.delete(vessel_shared_array_name)
            SharedArray.delete(seg_shared_array_name)
            
        return fundus_fnames, funduses, vessels, segs

    
class TrainBatchFetcher(SharedIterator):

    def __init__(self, train_fundus, train_vessel, train_seg, batch_size):
        self.train_fundus = np.array(train_fundus)
        self.train_vessel = np.array(train_vessel)
        self.train_seg = np.array(train_seg)
        self.batch_size = batch_size
        self.augment = True
        super(TrainBatchFetcher, self).__init__()
        
    def __call__(self):
        indices = np.array(range(len(self.train_fundus))) 
        random.shuffle(indices)
        fundus = self.train_fundus[indices]
        vessel = self.train_vessel[indices]
        seg = self.train_seg[indices]
        return super(TrainBatchFetcher, self).__call__(fundus, vessel, seg)
