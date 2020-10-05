#! /usr/bin/env python3
"""
Code Taken From: https://github.com/tensorflow/tensorflow/issues/20942 @kevintrankt commented on Jul 23, 2018
Code Taken from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
Wiki: Silhouette clustering -> https://en.wikipedia.org/wiki/Silhouette_(clustering)
Author: Yuval Kaneti⭐
"""

#### Imports ####
import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import silhouette_score
from integration.wrapers.base_wraper import BaseWraper
from integration.utils.data_utils import WraperOutput
from integration.utils.common import BATCH_SIZE_DEST

class KmeansTensorflowWraper(BaseWraper):
    """KmeansTensorflowWraper -> An implemntion of Kmeans & silhouette_score in Tensorflow."""
    def __init__(self, start_k: int, end_k: int, num_iteration: int, **kwargs: dict):
        """
        @param data: C{list} -> a list of Image Objects.
        @param start_k: C{int} -> the minimum number of Cluster to try kmeans with
        @param end_k: C{int} -> the maximum number of Cluster to try kmeans with
        """
        super().__init__(**kwargs)
        # self._data = data

        self._start_k = start_k
        self._end_k = end_k
        self._num_iteration = num_iteration
        self.kwargs = kwargs
        if self._loader.name != 'DataLoader': raise RuntimeError(f"{self._loader.name} is not supported with {self.name}, pleas use DataLoader -> --loader data_loader")
        # self.use_tensorboard = tensorboard

    def run(self):
        """
        @remarks *This is where the action starts.
                 *Will run kmeans (end_k - start_k) times and Compare the results with SilhouetteScore.
        """
        self._validate_input()
        (self.X, self.filenames) = self.Wraperize()
        self.wraper_output = self._silhouette()
        if self._use_tensorboard:
            # ?
            # self.update()
            self._tensorboard(dtype=self._loader.dtype, batch_size=self.kwargs.get(BATCH_SIZE_DEST))
        return self.wraper_output

    def _validate_input(self):
        """
        @remarks *Will just validate That the params are chill..
        """
        logging.debug("Validating Input for KmeansTensorflowWraper")
        assert self._start_k <= self._end_k 
        assert self._start_k > 1     
       
    def _silhouette(self):
        """
        @return C{SilhouetteScore} -> A WraperOutput object for TensorflowBaseWrapers.
        @remarks *will go over each number of clusters specified by start_k -> end_k.
                 *will Run kmeans against it & Calculate the silhouette_score.
                 *It will compare them and take the one it the highest Score.
        """
        score_list = []
        for n_clusters in range(self._start_k, self._end_k):
            logging.debug(f"Using: {n_clusters} Clusters")
            # Set num_clusters
            self._num_clusters = n_clusters
            cluster_labels = self.train()
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(self.X, cluster_labels)
                kmeans_wraper_output = KmeansWraperOutput(n_clusters=n_clusters, cluster_labels=cluster_labels, silhouette_avg=silhouette_avg)
                logging.info(kmeans_wraper_output)
                score_list.append(kmeans_wraper_output)
 
        score_list = sorted(score_list, key=lambda x: x.silhouette_avg, reverse=True)
        logging.info(f"{score_list[0].n_clusters} clusters had {score_list[0].silhouette_avg} Which is the best")

        return score_list[0]

    def train(self):
        """
        @remarks *I DONT UNDERSTAND ANYTHING HERE.
                 *See [https://stackoverflow.com/questions/33621643/how-would-i-implement-k-means-with-tensorflow],
                      [https://www.altoros.com/blog/using-k-means-clustering-in-tensorflow/],
                      For Details.   
        """
        assert self._num_clusters != None
        with tf.device('/GPU:0'):
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

            sess = tf.compat.v1.Session()
            sess.run(tf.compat.v1.global_variables_initializer())

            # 1000 iterations or no delta
            has_changed = True
            num_iter =  0
            while has_changed and num_iter < self._num_iteration:
                num_iter += 1
                has_changed, _ = sess.run([is_continue, loop])
            # see how the data is assigned
            res = sess.run(point_to_centroid_assignment)
            # print(list(res))
        return list(res)
    

class KmeansWraperOutput(WraperOutput):
    """KmeansWraperOutput -> A WraperOutput Object for KemansTensorflowWraper with Silhouette Score."""
    def __init__(self, n_clusters, cluster_labels, silhouette_avg):
        super().__init__(n_clusters, cluster_labels)
        self.silhouette_avg = silhouette_avg
    
    def __str__(self):
        return f"For n_clusters = {self.n_clusters}, The average silhouette_score is : {self.silhouette_avg}"