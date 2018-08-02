import multiprocessing
import threading
import Queue
from uuid import uuid4
import numpy as np
import SharedArray

import utils

img_h, img_w = 640, 640
n_grade_dr, n_grade_dme = 5, 3


def balance_per_class_indices(y, weights):
    weights = np.array(weights, dtype=float)
    p = np.zeros(len(y))
    for i, weight in enumerate(weights):
        p[y == i] = weight
    return np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                            p=np.array(p) / p.sum())


def load_shared(args):
    i, array_fundus_mean_subt_name, array_fundus_z_name, array_vessel_name, fundus, vessel, is_train, normalize = args
    array_fundus_mean_subt = SharedArray.attach(array_fundus_mean_subt_name)
    array_fundus_z = SharedArray.attach(array_fundus_z_name)
    array_vessel = SharedArray.attach(array_vessel_name)
    array_fundus_mean_subt[i], array_fundus_z[i], array_vessel[i] = utils.load_augmented_fundus_vessel([fundus], [vessel], is_train, normalize)

    
class BatchIterator(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, fundus, vessel, grade):
        self.fundus, self.vessel, self.grade = fundus, vessel, grade
        return self

    def __iter__(self):
        n_samples = self.fundus.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            fundus = self.fundus[sl]
            vessel = self.vessel[sl]
            grade = self.grade[sl]
            yield self.transform(fundus, vessel, grade)


class QueueIterator(BatchIterator):

    def __iter__(self):
        queue = Queue.Queue(maxsize=20)
        end_marker = object()

        def producer():
            for fundus_fns, fundus_mean_subt_img, fundus_z_img, vessel_img, grade in super(QueueIterator, self).__iter__():
                queue.put((np.array(fundus_fns), np.array(fundus_mean_subt_img), np.array(fundus_z_img), np.array(vessel_img), np.array(grade)))
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

    def transform(self, fundus, vessel, grade):
        shared_array_fundus_mean_subt_name = str(uuid4())
        shared_array_fundus_z_name = str(uuid4())
        shared_array_vessel_name = str(uuid4())
        
        try:
            shared_array_fundus_mean_subt = SharedArray.create(
                shared_array_fundus_mean_subt_name, [len(fundus), img_h, img_w, 3], dtype=np.float32)
            shared_array_fundus_z = SharedArray.create(
                shared_array_fundus_z_name, [len(fundus), img_h, img_w, 3], dtype=np.float32)
            shared_array_vessel = SharedArray.create(
                shared_array_vessel_name, [len(fundus), img_h, img_w, 1], dtype=np.float32)
            
            n_grades = len(grade)
            if self.grade_type == "DR":
                grade_onehot = np.zeros((n_grades, n_grade_dr))
            elif self.grade_type == "DME":
                grade_onehot = np.zeros((n_grades, n_grade_dme))
            for i in range(n_grades):
                grade_onehot[i, grade[i]] = 1
            
            args = []
            for i, _ in enumerate(fundus):
                args.append((i, shared_array_fundus_mean_subt_name, shared_array_fundus_z_name, shared_array_vessel_name, fundus[i], vessel[i], self.is_train, self.normalize))

            self.pool.map(load_shared, args)
            fundus_mean_subt_img = np.array(shared_array_fundus_mean_subt, dtype=np.float32)
            fundus_z_img = np.array(shared_array_fundus_z, dtype=np.float32)
            vessel_img = np.array(shared_array_vessel, dtype=np.float32)
        finally:
            SharedArray.delete(shared_array_fundus_mean_subt_name)
            SharedArray.delete(shared_array_fundus_z_name)
            SharedArray.delete(shared_array_vessel_name)

        return fundus, fundus_mean_subt_img, fundus_z_img, vessel_img, grade_onehot


class TrainBatchFetcher(SharedIterator):

    def __init__(self, training_set, batch_size, grade_type, class_weight, normalize):
        self.fundus, self.vessel, self.grade = training_set
        self.is_train = True
        self.grade_type = grade_type
        self.class_weight = class_weight
        self.normalize = normalize
        super(TrainBatchFetcher, self).__init__(batch_size)

    def __call__(self):
        indices = balance_per_class_indices(self.grade, weights=self.class_weight)
        fundus = self.fundus[indices]
        vessel = self.vessel[indices]
        grade = self.grade[indices]
        return super(TrainBatchFetcher, self).__call__(fundus, vessel, grade)


class ValidationBatchFetcher(SharedIterator):

    def __init__(self, validation_set, batch_size, grade_type, normalize):
        self.fundus, self.vessel, self.grade = validation_set
        self.is_train = False
        self.grade_type = grade_type
        self.normalize = normalize
        super(ValidationBatchFetcher, self).__init__(batch_size)

    def __call__(self):
        return super(ValidationBatchFetcher, self).__call__(self.fundus, self.vessel, self.grade)

