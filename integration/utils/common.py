#!/usr/bin/env python3
# Author: Yuval Kaneti

import os


#### CONSTANTS ####
WEIGHTS_FOLDER_PATH = "D:\\Datasets\\Imagefy\\weights"
IMAGE_SIZE = 124
IMAGE_TYPES = ['jpg', 'png', 'jpeg']
MAX_WORKERS = 25
OUTPUT_DIR_PATH = "D:\\Imagefy\\output"
VISUALIZATION_DIM = 3
MINI_KMEANS_NUM_ITERATIONS = 10
TENSORBOARD_LOG_DIR = 'logs'
CLUSTER_NAME_FORMAT = "cluster_{}"

DIR_DEST = 'dir'
VERBOSE_DEST = 'verbose'
SIZE_DEST = 'image_size'
TENSORBOARD_DEST = 'tensorboard'
TENSORBOARD_NAME_DEST = 'tensorboard_name'
KMEANS_DEST = 'kmeans'
START_DEST = 'start_k'
END_DEST = 'end_k'
MINI_KMEAND_DEST = 'mini-kmeans'
ITER_DEST = 'num_iterations'
NUM_CLUSTERS_DEST = 'num_clusters'
SAVE_MODEL_DEST = 'save'
WRAPER_PREFIX = '_tensorflow_wraper.py'
WRAPERS_BASE_DIR = 'wrapers'
WRAPERS_DIR = os.path.join(os.getcwd(), WRAPERS_BASE_DIR)

DATA_PARAM = 'data'
DATA_GENERATOR_PARAM = 'data_generator'
X_PARAM = 'X'
Y_PARAM = 'y'
WRAPER_PARAM = 'wraper'
TENSORBOARD_NAME_PARAM = None
# NUM_ITERATIONS_PARAM = 'num_iterations'
# NUM_CLUSTERS_PARAM = 'num_clusters'
