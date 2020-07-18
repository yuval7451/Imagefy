#!/usr/bin/env python3
# Author: Yuval Kaneti⭐
#### IMPROTS ####
import numpy as np
import tensorflow as tf

"""
##  with tf.device(self.device):
['ALL_DISTANCES', 'CLUSTER_CENTERS_VAR_NAME', 'CLUSTER_INDEX', 'COSINE_DISTANCE', 'KMEANS_PLUS_PLUS_INIT', 'RANDOM_INIT', 'SCORE', 
'SQUARED_EUCLIDEAN_DISTANCE', 'cluster_centers', 'config', 'eval_dir', 'evaluate', 'experimental_export_all_saved_models', 'export_saved_model',
'export_savedmodel', 'get_variable_names', 'get_variable_value', 'latest_checkpoint', 'model_dir', 'model_fn', 'params', 'predict', 
'predict_cluster_index', 'score', 'train', 'transform']
"""
k = 5
n = 100
variables = 2
points = np.random.uniform(0, 1000, [n, variables])
input_fn=lambda: tf.train.limit_epochs(tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)
with tf.device('/GPU:0'):
    kmeans=tf.contrib.factorization.KMeansClustering(num_clusters=k, use_mini_batch=False)
    previous_centers = None
    cluster_indices = None
    for _ in range(10):
        kmeans.train(input_fn)
        centers = kmeans.cluster_centers()
        if previous_centers is not None:
            print( 'delta:', centers - previous_centers)
            previous_centers = centers
            print ('score:', kmeans.score(input_fn))
            print ('centers:', centers)
            cluster_indices = list(kmeans.predict_cluster_index(input_fn))
            
        previous_centers = centers

for i, point in enumerate(points):
    cluster_index = cluster_indices[i]
    print ('point:', point, 'is in cluster', cluster_index, 'centered at', centers[cluster_index])
    
# modelPath = kmeans.export_savedmodel(export_dir_base="/path/",serving_input_receiver_fn=serving_input_receiver_fn)
# print(modelPath)