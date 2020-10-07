#!/usr/bin/env python3
# Author: Yuval Kaneti

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

#### CONSTANTS ####
WEIGHTS_FOLDER_PATH = "D:\\Datasets\\Imagefy\\weights"
IMAGE_SIZE = 224
MINI_KMEANS_NUM_EPOCHS = 10
MINI_KMEANS_BATCH_SIZE = 20
MAX_WORKERS = 50
IMAGE_TYPES = ['jpg', 'png', 'jpeg']
LOG_DIR = 'logs'
OUTPUT_DIR_PATH = "output"
CLUSTER_NAME_FORMAT = "cluster_{}"

DIR_DEST = 'dir_path'
VERBOSE_DEST = 'verbose'
SIZE_DEST = 'image_size'
TENSORBOARD_DEST = 'tensorboard'
MINI_KMEAND_DEST = 'mini_kmeans' 
INCEPTION_RESNET_DEST = 'inception'
EPOCHS_DEST = 'num_epochs'
BATCH_SIZE_DEST = 'batch_size'
NUM_CLUSTERS_DEST = 'num_clusters'
SAVE_MODEL_DEST = 'save'
NUM_ITERATION_DEST = 'num_iteration'
BASE_PATH_DEST = 'base_path'
MODEL_NAME_PARAM = 'model_name'
BASE_MODEL_DIR_PARAM = 'base_model_dir'

LOADER_DEST = 'loader'
TENSOR_LADER_DEST = 'tensor_loader'
LOADER_OPTIONS = [TENSOR_LADER_DEST]

WRAPERS_BASE_DIR = 'wrapers'
WRAPERS_DIR = os.path.join(os.getcwd(), WRAPERS_BASE_DIR)
WRAPER_PARAM = 'wraper'
DATA_PARAM = 'data'

X_PARAM = 'X'
Y_PARAM = 'y'
TOP_DEST = 'top'
TOP_PARAM = 3

WRAPER_PREFIX = '_tensorflow_wraper.py'
TENSORBOARD_NAME_PARAM = None

MINI_BATCH_KMEANS_TENSORFLOW_WRAPER = "MiniBatchKmeansTensorflowWraper"
INCEPTION_RESNET_TENSORFLOW_WRAPER = "InceptionResnetTensorflowWraper"
INCEPTION_RESNET_TENSORFLOW_WRAPER_OUTPUT = "InceptionResnetTensorflowWraperOutput"
INCEPTION_RESNET_INFERENCE_INPUT = "input_1"
PREDICTOR_INFERENCE_INPUTS = "predictor_inputs"
INCEPTION_RESNET_INFERENCE_DENSE = "dense"
OUTPUT_DIR_PATH_PARAM = 'output_dir_path'
IFERENCE_MODEL_DIR = "D:\\Imagefy\\resources\\models\\InceptionResNetV2\\inference\\1601923417"
EMBEDDINGS_TENSOR_NAME = "embeddings"