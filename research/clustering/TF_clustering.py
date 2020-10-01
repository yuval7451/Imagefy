#!/usr/bin/env python3
# Author: Yuval Kaneti⭐
#### IMPROTS ####
import os
import numpy as np
import tensorflow as tf
from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from utils.data_utils import load_data


k_class = 2

def TF_kmeans(data):
    input_fn=lambda: tf.train.limit_epochs(tf.convert_to_tensor(data, dtype=tf.float32), num_epochs=1)
    with tf.device('/GPU:0'):
        kmeans=tf.contrib.factorization.KMeansClustering(num_clusters=k_class, use_mini_batch=True)
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
                
            previous_centers = centers

    
    
    cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    for i, point in enumerate(points):
        cluster_index = cluster_indices[i]
        print ('point:', point, 'is in cluster', cluster_index, 'centered at', centers[cluster_index])
        
 
# modelPath = kmeans.export_savedmodel(export_dir_base="/path/",serving_input_receiver_fn=serving_input_receiver_fn)
# print(modelPath)


def main():
        TEST = True
        if TEST:
            PATH = os.path.join(DATA_FOLDER_PATH, TEST_IMAGES_FOLDER)
        else:
            PATH = os.path.join(DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER)

        data = load_data(PATH)
        print(data.shape)
        TF_kmeans(data)

if __name__ == '__main__':
    main()