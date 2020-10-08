#!/usr/bin/env python3
# Author: Yuval Kaneti

#### Imports ####
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

from imagefy.suits.base_suit import BaseSuit
from imagefy.utils.data_utils import IOWraper, TensorLoader
from imagefy.utils.common import  BASE_PATH_DEST, DATA_PARAM, TOP_DEST, LOADER_DEST
from imagefy.wrapers.mini_batch_kmeans_tensorflow_wraper import MiniBatchKmeansTensorflowWraper
from imagefy.wrapers.inception_resnet_tensorflow_wraper import InceptionResnetTensorflowWraper

class IntergrationSuit(BaseSuit):
    """IntergrationSuit -> Some Kind of Class that controls everything."""
    def __init__(self, **kwargs: dict):
        """
        @param kwargs: C{dict} -> A dict with all parameters passed on Runtime.
        @remarks *Curently the bos of the module.
        """
        super().__init__(**kwargs)
        # tf.profiler.experimental.start(self.base_model_dir)


    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseLoader, @BaseWraper & @IOWraper
        """
        # Load Data
        self._loader = TensorLoader(**self.kwargs)  
        self.data = self._loader.run()
        self.kwargs.update({DATA_PARAM: self.data, LOADER_DEST: self._loader})
       
       # Kmeans
        self.kmeans = MiniBatchKmeansTensorflowWraper(**self.kwargs) 
        self.WraperOutput = self.kmeans.run()
        
        # Handle IO And File Transfering
        self.IOHandler = IOWraper(kmeans_data=self.data, wraper_output=self.WraperOutput, model_name=self.model_name, base_path=self.kwargs.get(BASE_PATH_DEST))
        self.IOHandler.create_output_dirs()
        self.IOHandler.merge_kmeans_data()
        self.IOHandler.move_kmeans_data()
        logging.info(f"Check {self.base_model_dir} for output")

        # Inception
        self.inception = InceptionResnetTensorflowWraper(**self.kwargs)
        inception_data = self._loader.create_inception_data()
        self.WraperOutput = self.inception.run()
        self.IOHandler.set_inception_data(inception_data=inception_data, wraper_output=self.WraperOutput)

        self.IOHandler.merge_inception_data(self.kwargs.get(TOP_DEST))
        self.IOHandler.move_inception_data()

        # tf.profiler.experimental.stop()
        logging.info("Finished running Suit")
