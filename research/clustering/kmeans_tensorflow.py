#! /usr/bin/env python3
"""
Code Taken From: https://github.com/tensorflow/tensorflow/issues/20942 @kevintrankt commented on Jul 23, 2018
Author: Yuval Kaneti⭐
"""
import numpy as np
import tensorflow as tf

import os
from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from utils.data_utils import load_data

class KmeansTensorflow:
    """
    """
    def __init__(self, input_matrix, num_clusters):
        """
        """
        self._input = input_matrix
        self._num_clusters = num_clusters


    def train(self):
        """
        """
        k = self._num_clusters
        # centroid initialization
        start_pos = tf.Variable(self._input[np.random.randint(self._input.shape[0], size=k), :],
                                dtype=tf.float32)
        centroids = tf.Variable(start_pos.initialized_value(), 'S', dtype=tf.float32)

        # populate points
        points = tf.Variable(self._input, 'X', dtype=tf.float32)
        ones_like = tf.ones((points.get_shape()[0], 1))
        prev_assignments = tf.Variable(tf.zeros((points.get_shape()[0],), dtype=tf.int64))

        # distance function
        p1 = tf.matmul(
            tf.expand_dims(tf.reduce_sum(tf.square(points), 1), 1),
            tf.ones(shape=(1, k))
        )
        p2 = tf.transpose(tf.matmul(
            tf.reshape(tf.reduce_sum(tf.square(centroids), 1), shape=[-1, 1]),
            ones_like,
            transpose_b=True
        ))
        distance = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(points, centroids, transpose_b=True))

        # assign each point to a closest centroid
        point_to_centroid_assignment = tf.argmin(distance, axis=1)

        # recalculate the centroid (mean)
        total = tf.unsorted_segment_sum(points, point_to_centroid_assignment, k)
        count = tf.unsorted_segment_sum(ones_like, point_to_centroid_assignment, k)
        means = total / count

        # continue if there is any delta
        is_continue = tf.reduce_any(tf.not_equal(point_to_centroid_assignment, prev_assignments))

        with tf.control_dependencies([is_continue]):
            loop = tf.group(centroids.assign(means), prev_assignments.assign(point_to_centroid_assignment))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # 1000 iterations or no delta
        has_changed, cnt = True, 0
        while has_changed and cnt < 1000:
            cnt += 1
            has_changed, _ = sess.run([is_continue, loop])
        # see how the data is assigned
        res = sess.run(point_to_centroid_assignment)
        # print(list(res))
        return list(res)
        
def main():
    TEST = True
    if TEST:
        PATH = os.path.join(DATA_FOLDER_PATH, TEST_IMAGES_FOLDER)
    else:
        PATH = os.path.join(DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER)

    data = load_data(PATH,resize=True, size=224)
    print(data.shape)
    
    TFKMeans = KmeansTensorflow(data, 2)
    TFKMeans.train()

    
    
if __name__ == '__main__':
    main()