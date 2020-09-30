#! /usr/bin/env python3
"""
Code Taken From: https://github.com/tensorflow/tensorflow/issues/20942 @kevintrankt commented on Jul 23, 2018
Code Taken from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
Wiki: Silhouette clustering -> https://en.wikipedia.org/wiki/Silhouette_(clustering)
Author: Yuval Kaneti⭐
"""
import numpy as np
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)

from development.base_wraper import BaseWraper
from sklearn.metrics import silhouette_samples, silhouette_score


import os
from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER, IMAGE_SIZE
from utils.data_utils import load_data

class KmeansTensorflowWraper(BaseWraper):
    """
    """
    def __init__(self, data, start_k, stop_k, verbose):
        """
        """
        self._data = data
        self._start_k = start_k
        self._stop_k = stop_k
        self._verbose = verbose
        
    def start(self):
        """
        """
        logging.info("Starting KmeansTensorflowWraper")
        self._validate_input()
        (self.X, self.Y) = self._process_input()
        cluster_labels = self._silhouette() 
        regrouped_data = self._cluster_labels_to_data(cluster_labels)
        return regrouped_data
        
    def _process_input(self):
            x = np.array([tup[1].flatten() for tup in self._data[0]])
            y = [tup[0] for tup in self._data[0]]
            return (x, y)

    def _validate_input(self):
        """
        """
        assert self._start_k < self._stop_k 
        assert self._start_k > 1     
    
    def _cluster_labels_to_data(self, cluster_labels):
        """
        """
        classes_dict = {}
        data = []
        for index, image_class in enumerate(cluster_labels):
            image = self.X[index].reshape((IMAGE_SIZE, IMAGE_SIZE, 3))
            image_path = self.Y[index]
            if image_class in classes_dict.keys():
                classes_dict[image_class].append((image_path, np.expand_dims(image, axis=0)))
            else:
                classes_dict[image_class] = [(image_path, np.expand_dims(image, axis=0))]
                
        for key in list(classes_dict.keys()):
            data.append(classes_dict[key])    
        
        return data
    
    def _silhouette(self):
        score_list = []
        for n_clusters in range(self._start_k, self._stop_k):
            if self._verbose:
                logging.info(f"Using: {n_clusters}")
            # Set num_clusters
            self._num_clusters = n_clusters
            cluster_labels = self.train()
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(self.X, cluster_labels)
                if self._verbose:
                    logging.debug("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
                score_list.append((n_clusters, silhouette_avg, cluster_labels))

                
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        if self._verbose:
            logging.debug(20*"-")
            logging.debug(f"{score_list[0][0]} clusters had {score_list[0][1]} Which is the best")
            logging.debug(20*"-")

        return score_list[0][2]

    def train(self):
        """
        """
        assert self._num_clusters != None
        # centroid initialization
        start_pos = tf.Variable(self.X[np.random.randint(self.X.shape[0], size=self._num_clusters), :],
                                dtype=tf.float32)
        centroids = tf.Variable(start_pos.initialized_value(), 'S', dtype=tf.float32)

        # populate points
        points = tf.Variable(self.X, 'X', dtype=tf.float32)
        ones_like = tf.ones((points.get_shape()[0], 1))
        prev_assignments = tf.Variable(tf.zeros((points.get_shape()[0],), dtype=tf.int64))

        # distance function
        p1 = tf.matmul(
            tf.expand_dims(tf.reduce_sum(tf.square(points), 1), 1),
            tf.ones(shape=(1, self._num_clusters))
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
        total = tf.unsorted_segment_sum(points, point_to_centroid_assignment, self._num_clusters)
        count = tf.unsorted_segment_sum(ones_like, point_to_centroid_assignment, self._num_clusters)
        means = total / count

        # continue if there is any delta
        is_continue = tf.reduce_any(tf.not_equal(point_to_centroid_assignment, prev_assignments))

        with tf.control_dependencies([is_continue]):
            loop = tf.group(centroids.assign(means), prev_assignments.assign(point_to_centroid_assignment))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # 1000 iterations or no delta
        has_changed = True
        num_iter =  0
        while has_changed and num_iter < 1000:
            num_iter += 1
            has_changed, _ = sess.run([is_continue, loop])
        # see how the data is assigned
        res = sess.run(point_to_centroid_assignment)
        # print(list(res))
        return list(res)
        
# def main():
#     TEST = True
#     if TEST:
#         PATH = os.path.join(DATA_FOLDER_PATH, TEST_IMAGES_FOLDER)
#     else:
#         PATH = os.path.join(DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER)

#     data = load_data(PATH,resize=True, size=224)
#     print(data.shape)
    
#     TFKMeans = KmeansTensorflowWraper(data, 2, 6, True)
#     labels = TFKMeans.start()
#     print(labels)
       
# if __name__ == '__main__':
#     main()