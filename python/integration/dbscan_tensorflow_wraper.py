#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import numpy as np
import tensorflow as tf; tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from integration.base_wraper import BaseWraper
from integration.data_utils import BaseScore


class DbscanTensorflowWraper(BaseWraper):
    """
    """
    def __init__(self, data, epsilon=4, min_points=4):
        """
        """
        self._data = data
        (self.X, self.Y) = self._process_input()
        print(self.X.shape)
        self._eps = epsilon
        self._min_pts = min_points

    def _process_input(self):
        """
        """
        x = np.asarray([ImageObj.data for ImageObj in self._data])
        y = [ImageObj.src_path for ImageObj in self._data]
        return (x, y)

    def run(self):
        """
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

        with tf.Session() as sess:
            results = (sess.run(final_labels, feed_dict={in_set:self.X})).reshape((1, -1))

        results = results.reshape((-1, 1))

        return results


