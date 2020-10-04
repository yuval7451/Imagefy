#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import os
import logging
import numpy as np
import tensorflow as tf
from integration.wrapers.base_wraper import BaseWraper
from integration.utils.data_utils import WraperOutput

class DbscanTensorflowWraper(BaseWraper):
    """DbscanTensorflowWraper -> An implemention of DBSCAN using Tensorflow API."""
    def __init__(self, data : np.ndarray, epsilon: int=4, min_points : int=4):
        """
        @param data: C{np.ndarray} -> The Flattened Data. Loaded Via DataLoaderWraper.
        @param epsilon: C{int} -> Some parameter it needs, Read: [https://en.wikipedia.org/wiki/DBSCAN].
        @param min_points: C{int} -> The Min number of point in each Cluster.
        @remarks *DOES NOT WORK WITH TENSORFLOW 1.5.
        """
        raise NotImplementedError("DOES NOT WORK WITH TENSORFLOW 1.5 Yet..")
        self._data = data
        (self.X, self.y) = self.Wraperize()
        logging.debug(f"Data Shape: {self.X.shape}")
        self._eps = epsilon
        self._min_pts = min_points



    def run(self):
        """
        @return No Idea??
        @remarks *This is were the Actions Starts.
                 *See [https://stackoverflow.com/questions/49934606/how-to-implement-dbscan-clustering-in-tensorflow]
        """
        def merge_core_points_into_clusters(elems):
            """
            """
            row = elems
            mat = core_points_connection_matrix
            nonempty_intersection_inds = tf.where(tf.reduce_any(tf.logical_and(row, mat), axis=1))
            cumul = tf.logical_or(row, mat)
            subcumul = tf.gather_nd(cumul, nonempty_intersection_inds)
            return tf.reduce_any(subcumul, axis=0)

        def label_clusters(elems):
            """
            """
            return tf.reduce_min(tf.where(elems))

        def get_subsets_for_labels(elems):
            """
            """
            val = elems[0]
            labels = elems[1]
            conn = relation_matrix

            inds = tf.where(tf.equal(labels, val))
            masks = tf.gather_nd(conn, inds)
            return tf.reduce_any(masks, axis=0)

        def scatter_labels(elems):
            label = tf.expand_dims(elems[0], 0)
            mask = elems[1]
            return label*tf.cast(mask, dtype=tf.int64)

        in_set = tf.placeholder(tf.float64)
        # distance matrix
        r = tf.reduce_sum(in_set*in_set, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        dist_mat = tf.sqrt(r - 2*tf.matmul(in_set, tf.transpose(in_set)) + tf.transpose(r))

        # for every point show, which points are within eps distance of that point (including that point)
        relation_matrix = dist_mat <= self._eps

        # number of points within eps-ball for each point
        num_neighbors = tf.reduce_sum(tf.cast(relation_matrix, tf.int64), axis=1)

        # for each point show, whether this point is core point
        core_points_mask = num_neighbors >= self._min_pts

        # indices of core points
        core_points_indices = tf.where(core_points_mask)

        core_points_connection_matrix = tf.cast(core_points_mask, dtype=tf.int64) * tf.cast(relation_matrix, dtype=tf.int64)
        core_points_connection_matrix = tf.cast(core_points_connection_matrix, dtype=tf.bool)
        core_points_connection_matrix = tf.logical_and(core_points_connection_matrix, core_points_mask)

        merged = tf.map_fn(
            merge_core_points_into_clusters,
            core_points_connection_matrix,
            dtype=tf.bool
        )

        nonempty_clusters_records = tf.gather_nd(merged, core_points_indices)

        marked_core_points = tf.map_fn(label_clusters, nonempty_clusters_records, dtype=tf.int64)

        _, labels_core_points = tf.unique(marked_core_points, out_idx=tf.int64)

        labels_core_points = labels_core_points+1

        unique_labels, _ = tf.unique(labels_core_points)

        labels_all = tf.scatter_nd(
            tf.cast(core_points_indices, tf.int64),
            labels_core_points,
            shape=tf.cast(tf.shape(core_points_mask), tf.int64)
        )

        # for each label return mask, which points should have this label
        ul_shape = tf.shape(unique_labels)
        labels_tiled = tf.maximum(tf.zeros([ul_shape[0], 1], dtype=tf.int64), labels_all)

        labels_subsets = tf.map_fn(
            get_subsets_for_labels,
            (unique_labels, labels_tiled),
            dtype=tf.bool
        )

        final_labels = tf.map_fn(
            scatter_labels,
            elems=(tf.expand_dims(unique_labels, 1), labels_subsets),
            dtype=tf.int64
        )

        final_labels = tf.reduce_max(final_labels, axis=0)

        with tf.compat.v1.Session() as sess:
            results = (sess.run(final_labels, feed_dict={in_set:self.X})).reshape((1, -1))

        results = results.reshape((-1, 1))

        return results

