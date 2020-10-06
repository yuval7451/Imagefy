#!/usr/bin/env python3
# Author: Yuval Kaneti

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

#### CONSTANTS ####
WEIGHTS_FOLDER_PATH = "D:\\Datasets\\Imagefy\\weights"
IMAGE_SIZE = 124
IMAGE_TYPES = ['jpg', 'png', 'jpeg']
MAX_WORKERS = 25
OUTPUT_DIR_PATH = "output"
VISUALIZATION_DIM = 3
MINI_KMEANS_NUM_EPOCHS = 10
MINI_KMEANS_BATCH_SIZE = 20
LOG_DIR = 'logs'
MODEL_LOG_DIR = 'model'
TENSORBOARD_LOG_DIR = 'tensorboard'
CLUSTER_NAME_FORMAT = "cluster_{}"

DIR_DEST = 'dir_path'
VERBOSE_DEST = 'verbose'
SIZE_DEST = 'image_size'
TENSORBOARD_DEST = 'tensorboard'
TENSORBOARD_NAME_DEST = 'tensorboard_name'
KMEANS_DEST = 'kmeans'
START_DEST = 'start_k'
END_DEST = 'end_k'
MINI_KMEAND_DEST = 'mini_kmeans' #FIXME!!!
EPOCHS_DEST = 'num_epochs'
BATCH_SIZE_DEST = 'batch_size'
NUM_CLUSTERS_DEST = 'num_clusters'
SAVE_MODEL_DEST = 'save'
NUM_ITERATION_DEST = 'num_iteration'
BASE_PATH_DEST = 'base_path'

LOADER_DEST = 'loader'
DATA_LOADER_DEST = 'data_loader'
TENSOR_LADER_DEST = 'tensor_loader'
LOADER_OPTIONS = [DATA_LOADER_DEST, TENSOR_LADER_DEST]

WRAPERS_BASE_DIR = 'wrapers'
WRAPERS_DIR = os.path.join(os.getcwd(), WRAPERS_BASE_DIR)
WRAPER_PARAM = 'wraper'
DATA_PARAM = 'data'

X_PARAM = 'X'
Y_PARAM = 'y'

WRAPER_PREFIX = '_tensorflow_wraper.py'
TENSORBOARD_NAME_PARAM = None

MINI_BATCH_KMEANS_TENSORFLOW_WRAPER = "MiniBatchKmeansTensorflowWraper"
KMEANS_TENSORFLOW_WRAPER = "KmeansTensorflowWraper"
INCEPTION_RESNET_TENSORFLOW_WRAPER = "InceptionResnetWraper"
DATASET_DTYPE = tf.data.Dataset

# NUM_EPOCHS_PARAM = 'NUM_EPOCHS'
# NUM_CLUSTERS_PARAM = 'num_clusters'


# NUM_EPOCHS