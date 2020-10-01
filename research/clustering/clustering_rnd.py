#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### IMPROTS ####
import os
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from utils.data_utils import load_data


#### FUNCTIONS ####
def KMEANS_cluster(data, n_digits = 2):
    estimator = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(data)
    labels = estimator.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    return estimator

def DBSCAN_cluster(data):
    estimator = DBSCAN(eps=0.3, min_samples=10).fit(data)
    labels = estimator.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    return estimator



def main():
        TEST = False
        if TEST:
            PATH = os.path.join(DATA_FOLDER_PATH, TEST_IMAGES_FOLDER)
        else:
            PATH = os.path.join(DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER)

        data = load_data(PATH)
        print(data.shape)
        estimator = DBSCAN_cluster(data)

if __name__ == '__main__':
    main()