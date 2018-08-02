import multiprocessing
import threading
import Queue
from uuid import uuid4
import os
import numpy as np
import SharedArray

import utils
import random

img_h, img_w = 640, 640


def load_shared(args):
    i, array_fundus_name, array_vessel_name, array_lm_name, fundus, vessel, lm, is_train = args
    array_fundus = SharedArray.attach(array_fundus_name)
    array_vessel = SharedArray.attach(array_vessel_name)
    array_lm = SharedArray.attach(array_lm_name)
    array_fundus[i], array_vessel[i], array_lm[i] = utils.load_augmented_lm(fundus, vessel, lm, augment=is_train)
    
class BatchIterator(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, fundus, vessel, coords):
        self.fundus, self.vessel, self.coords = fundus, vessel, coords
        return self

    def __iter__(self):
        n_samples = self.fundus.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs - 1):
            sl = slice(i * bs, (i + 1) * bs)
            fundus = self.fundus[sl]
            vessel = self.vessel[sl]
            coords = self.coords[sl]
            yield self.transform(fundus, vessel, coords)


class QueueIterator(BatchIterator):

    def __iter__(self):
        queue = Queue.Queue(maxsize=20)
        end_marker = object()

        def producer():
            for fundus_img, vessel_img, coord, fundus in super(QueueIterator, self).__iter__():
                queue.put((np.array(fundus_img), np.array(vessel_img), np.array(coord), np.array(fundus)))
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

    def __init__(self, *args):
        self.pool = multiprocessing.Pool()
        super(SharedIterator, self).__init__(*args)

    def transform(self, fundus, vessel, coords):
        shared_array_fundus_name = str(uuid4())
        shared_array_vessel_name = str(uuid4())
        shared_array_lm_name = str(uuid4())
        try:
            shared_array_fundus = SharedArray.create(
                shared_array_fundus_name, [len(fundus), img_h, img_w, 3], dtype=np.float32)
            shared_array_vessel = SharedArray.create(
                shared_array_vessel_name, [len(fundus), img_h, img_w, 1], dtype=np.float32)
            shared_array_lm = SharedArray.create(
                shared_array_lm_name, [len(fundus), 4], dtype=np.float32)
                                        
            args = []
            
            for i, fname in enumerate(fundus):
                args.append((i, shared_array_fundus_name, shared_array_vessel_name, shared_array_lm_name, fundus[i], vessel[i], coords[i], self.is_train))

            self.pool.map(load_shared, args)
            fundus_img = np.array(shared_array_fundus, dtype=np.float32)
            vessel_img = np.array(shared_array_vessel, dtype=np.float32)
            coords_arr = np.array(shared_array_lm, dtype=np.float32)
        finally:
            SharedArray.delete(shared_array_fundus_name)
            SharedArray.delete(shared_array_vessel_name)
            SharedArray.delete(shared_array_lm_name)

        return fundus_img, vessel_img, coords_arr, fundus 


class TrainBatchFetcher(SharedIterator):

    def __init__(self, fundus_dir, vessel_dir, od_label_path, fovea_label_path, batch_size):
        self.fundus, self.vessel, self.coords = utils.load_coords(fundus_dir, vessel_dir, od_label_path, fovea_label_path)
        self.is_train = True
        super(TrainBatchFetcher, self).__init__(batch_size)

    def __call__(self):
        indices = np.arange(len(self.fundus))
        random.shuffle(indices)
        fundus = self.fundus[indices]
        vessel = self.vessel[indices]
        coords = self.coords[indices]
        return super(TrainBatchFetcher, self).__call__(fundus, vessel, coords)


class ValidationBatchFetcher(SharedIterator):

    def __init__(self, fundus_dir, vessel_dir, od_label_path, fovea_label_path, batch_size):
        self.fundus, self.vessel, self.coords = utils.load_coords(fundus_dir, vessel_dir, od_label_path, fovea_label_path)
        self.is_train = False
        super(ValidationBatchFetcher, self).__init__(batch_size)

    def __call__(self):
        return super(ValidationBatchFetcher, self).__call__(self.fundus, self.vessel, self.coords)

