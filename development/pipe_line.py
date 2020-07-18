#!/usr/bin/env python3
# Author: Yuval Kaneti⭐
#### IMPROTS ####
import os
import numpy as np
import tensorflow as tf
# import argparse
from Development.inception_resnet_wraper import InceptionResnetWraper
from Development.kmeans_tensorflow_wraper import KmeansTensorflowWraper
from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from utils.data_utils import load_data, pipeline_to_cluster

class Pipeline(object):
    """
    """
    # OPTIONS = 
    def __init__(self, dir_path : str, mode : str):
        """
        """
        self.dir_path = dir_path
        self.mode = mode
        
    def _load_data(self):
        """
        """
        return load_data_pipeline(self.dir_path)