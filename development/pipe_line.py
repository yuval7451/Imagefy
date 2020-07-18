#!/usr/bin/env python3
# Author: Yuval Kaneti⭐
#### IMPROTS ####
##TODO: TEST
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
    def _load_data(self):
        """
        """
        return load_data(self.dir_path)
    
    
    def _full_pipeline(self):
        """
        """
        self.kmeans = KmeansTensorflowWraper(self.data)
        k_start = 2
        k_stop = 5 #len(self.data)
        self.inception = InceptionResnetWraper(gpu=True)
        image_classes = self.kmeans.silhouette(k_start, k_stop, verbose=True)
        constract_classes = self._constract_classes(image_classes)
        total_scores = []
        for image_class in constract_classes:
            tensor_class = pipeline_to_tensor(image_class)
            scores = self.inception.predict(tensor_class)
            ##FIXME: 
            total_scores.append(scores[:3])
        
        print(total_scores)
        
    def _constract_classes(self, image_classes):
        """
        """
        constracted_classes = {}
        classes_list = []
        for index, image_class in enumerate(image_classes):
            image = self.data[index]
            if image_class in constracted_classes.keys():
                constracted_classes[image_class].append(image)
            else:
                constracted_classes[image_class] = [image]
                
        for key in list(constracted_classes.keys()):
            classes_list.append(constracted_classes[key])    
        
        return classes_list