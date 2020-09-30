#!/usr/bin/env python3
# Author: Yuval Kaneti⭐
#### IMPROTS ####
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# import numpy as np
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
from development.inception_resnet_wraper import InceptionResnetWraper
from development.kmeans_tensorflow_wraper import KmeansTensorflowWraper
from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from utils.data_utils import load_data
from development.argument_parser import arg_parser, validate_parse
class Pipeline(object):
    """
    """
    def __init__(self):
        """
        """
        self.options = {
            "all" : self._full_pipeline,
            "inception" : self._inception,
            "kmeans" : self._cluster
        }
        
        args = arg_parser()
        validate_parse(args)
        self.dir_path = str(args.dir)
        self._verbose = "true" == args.verbose
        self.resize = "true" == args.resize
        self.gpu = "true" == args.gpu
        self.mode = str(args.mode)
        if self.mode != 'inception':
            self.start_k = int(args.start)
            self.stop_k = int(args.stop)

    def start(self):
        self.data = self._load_data()
        return_val = self.options[self.mode]()
        # print(return_val)
        return return_val
    
    
    def _load_data(self):
        """
        """
        return load_data(self.dir_path, resize=self.resize)
    

    def _full_pipeline(self):
        """
        """
        regroupd_data = self._cluster() 
        score_lists = self._inception(regroupd_data=regroupd_data)
        return score_lists
    

    def _inception(self, regroupd_data = None):
        """
        """
        if regroupd_data:
            Inception = InceptionResnetWraper(data=regroupd_data, gpu=self.gpu, verbose=self._verbose)
        else:
            Inception = InceptionResnetWraper(data=self.data, gpu=self.gpu, verbose=self._verbose)
        score_lists = Inception.start()
        return score_lists
    
    def _cluster(self):
        """
        """
        kmeans = KmeansTensorflowWraper(data=self.data, start_k=self.start_k, stop_k=self.stop_k, verbose=self._verbose)
        regroupd_data = kmeans.start()
        return regroupd_data
    
    

def main():
    pp = Pipeline()
    pp.start()

if __name__ == '__main__':
    main()