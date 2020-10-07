#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####

import os
import logging
import numpy as np
import tensorflow as tf
from imagefy.wrapers.base_wraper import BaseWraper
from imagefy.utils.data_utils import WraperOutput


class MeanShiftTensorflowWraper(BaseWraper):
    """MeanShiftTensorflowWraper -> An implemntion of mean-shift in Tensorflow."""
    def __init__(self, data : np.ndarray, radius : int=None, chunk_size : int=None):
        """
        @param data: C{np.ndarray} -> a List of Image Objects
        @param radius: C{int} -> the radius for Clustering.
        @param chunk_size: C{int} -> How many points to look at, at the same time.
        """
        raise NotImplementedError(f"Mean Shit is Curently not Suported.")
        self._data = data
        (self.X, self.y) = self.Wraperize()

        self.radius = radius
        self.chunk_size= chunk_size

        self.radius_t = tf.placeholder(shape=(), dtype=tf.float32)
        self.orig_data_all_t = tf.placeholder(shape=(None, None), dtype=tf.float32)
        
        self.indices_t = tf.placeholder(shape=(None,), dtype=tf.int32)
        data_batch_t = tf.gather(self.orig_data_all_t, self.indices_t)
        self.is_near_path_init = tf.zeros(
            shape=(tf.shape(data_batch_t)[0], tf.shape(self.orig_data_all_t)[0]), dtype=tf.bool)
        
        self.converged_data_batch_t, _, self.is_near_path_t = tf.while_loop(
            cond=self.has_not_converged_t, 
            body=self.mean_shift_step_t,
            loop_vars=(data_batch_t, np.float32(np.inf), self.is_near_path_init))
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
    
    def run(self):
        """
        @return C{DBScanScore} -> A WraperOutput Object returned By TensorflowBaseWraper's.
        @remarks *will run self.apply with the data.
        """
        # Now apply 
        peaks, labels = self.apply(self.X, self.radius, self.chunk_size)
        logging.info("Finished Applying Mean-Shift")
        return DBScanScore(cluster_labels=labels)

    def _process_input(self):
        """
        @return C{tuple} -> A Tuple Containing X,y.
        @remarks *An Halper function to Convert From Image to X(np.ndarray) & y(list).
                 *Should be implemented in the Image object or BaseWraper.
        """
        x = np.asarray([ImageObj.data for ImageObj in self._data])
        y = [ImageObj.src_path for ImageObj in self._data]
        return (x, y)
        
    def has_not_converged_t(self, data_batch_t, max_shift_t, is_near_path_t):
        return max_shift_t > 0.1
    
    def mean_shift_step_t(self, data_t, max_shift_t, is_near_path_t):
        """
        @remarks *Some stuff I Dont Understand.
        """
        data1_t = tf.expand_dims(data_t, 2) # shape [m, d, 1]
        data2_t = tf.transpose(tf.expand_dims(self.orig_data_all_t, 2), [2, 1, 0]) # shape [1, d, n]
        
        dist_t = tf.norm(data1_t - data2_t, axis=1, keepdims=True) # shape [m, 1, n]
        
        is_within_radius_t = tf.cast(dist_t <= self.radius_t, tf.float32) + tf.zeros_like(data2_t) # shape [m, d, n]
        num_within_radius_t = tf.count_nonzero(is_within_radius_t, axis=2, dtype=data_t.dtype)  # shape [m, d]
        
        data_within_radius_t = is_within_radius_t * data2_t # shape [m, d, n]
        sum_within_radius_t = tf.reduce_sum(data_within_radius_t, axis=2)  # shape [m, d]
        shifted_data_t = sum_within_radius_t / num_within_radius_t

        shift_t = tf.norm(shifted_data_t - data_t, axis=1)
        max_shift_t = tf.reduce_max(shift_t)
        
        is_near_path_new_t = tf.logical_or(tf.squeeze(dist_t, 1) <= self.radius_t/2, is_near_path_t) # shape [m, n]
        return shifted_data_t, max_shift_t, is_near_path_new_t
    
    def apply(self, data, radius, chunk_size):
        """
        @return C{tuple} -> The Peaks(np.ndarray) & Labels(np.ndarray)
        @remarks *Applying Mean-Shift To the Data.
        """
        labels = np.full(len(data), fill_value=-1, dtype=int)
        peaks = np.empty_like(data)
        thres_sq = (radius/2) ** 2
        n_peaks = 0
       
        while np.any(labels==-1):
            indices = np.argwhere(labels==-1)[:chunk_size, 0]
            
            converged_data, is_near_path = self.sess.run(
                [self.converged_data_batch_t, self.is_near_path_t], 
                feed_dict={
                    self.orig_data_all_t: data.astype(np.float32),
                    self.radius_t: np.float32(radius),
                    self.indices_t: np.int32(indices)})
            
            for idx, peak, is_near_this_path in zip(indices, converged_data, is_near_path):
                label = None

                # Compare found peak to existing peaks
                if n_peaks > 0:
                    dist = np.linalg.norm(peaks[:n_peaks] - peak, axis=1)
                    label_of_nearest_peak = np.argmin(dist)
                    logging.debug(f"Distance to nearest peak is: {dist[label_of_nearest_peak]}")

                    # If the nearest existing peak is near enough, take its label
                    if dist[label_of_nearest_peak] <= radius/2:
                        label = label_of_nearest_peak
                        logging.debug(f"label: {label}")
                # No existing peak was near enough, create new one
                if label is None:
                    label = n_peaks
                    peaks[label] = peak
                    n_peaks += 1
                    
                    dist_sq = np.linalg.norm(data - peak, axis=1)
                    labels[dist_sq <= radius] = label

                labels[is_near_this_path] = label

        return peaks[:n_peaks], labels
        
class DBScanScore(WraperOutput):
    """ DBScanScore -> A WraperOutput Object for MeanShiftTensorflowWraper."""
    def __init__(self, cluster_labels):
        n_clusters = max(cluster_labels) + 1
        super().__init__(n_clusters, cluster_labels)

    def __str__(self):
        return f"There are {self.n_clusters} Clusters"
