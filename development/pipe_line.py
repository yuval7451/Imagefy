#!/usr/bin/env python3
# Author: Yuval Kaneti⭐
#### IMPROTS ####
##TODO: TEST
import os
import numpy as np
import tensorflow as tf
# import argparse
from development.inception_resnet_wraper import InceptionResnetWraper
from development.kmeans_tensorflow_wraper import KmeansTensorflowWraper
from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from utils.data_utils import load_data, pipeline_to_cluster

class Pipeline(object):
    """
    """
    def __init__(self, dir_path : str, mode : str):
        """
        """
        self.dir_path = dir_path
        self.mode = mode
        self.data = self._load_data()
        self.options = {
            "ALL" : self._full_pipeline,
            "NN" : self._nn,
            "DIR" : self._dir
        }
        self.top = 3
        self._verbose = True
    def _load_data(self):
        """
        """
        return load_data(self.dir_path)
    
