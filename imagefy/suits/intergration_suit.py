#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
from imagefy.suits.base_suit import BaseSuit
from imagefy.utils.data_utils import IOWraper, TensorLoader
from imagefy.utils.common import  BASE_PATH_DEST, OUTPUT_DIR_PATH, DATA_PARAM, TOP_DEST, VERBOSE_DEST, WRAPER_PARAM, LOADER_DEST, \
    MODEL_NAME_PARAM, BASE_MODEL_DIR_PARAM, OUTPUT_DIR_PATH_PARAM
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
        # Asign Parameters
        self.kwargs = kwargs
        # Verbosity
        self.verbose = self.kwargs.get(VERBOSE_DEST, False)
        logging.getLogger().setLevel(logging.DEBUG if self.verbose else logging.INFO)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO if self.verbose else tf.compat.v1.logging.ERROR)

        self.WraperOutput = None
        self.IOHandler = None

        self.kwargs.update({MODEL_NAME_PARAM: self.model_name, 
                            BASE_PATH_DEST: self.base_path, 
                            BASE_MODEL_DIR_PARAM: self.base_model_dir,
                            OUTPUT_DIR_PATH_PARAM: self.output_dir_path})

        logging.debug(str(self.kwargs))

    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseLoader, @BaseWraper & @IOWraper
        """
        # Load Data
        self._loader = TensorLoader(**self.kwargs)  
        self.data = self._loader.run()
        self.kwargs.update({DATA_PARAM: self.data, LOADER_DEST: self._loader})
       
        self.kmeans = MiniBatchKmeansTensorflowWraper(**self.kwargs) 
        self.inception = InceptionResnetTensorflowWraper(**self.kwargs)

        self.WraperOutput = self.kmeans.run()
        
        # Handle IO And File Transfering
        self.IOHandler = IOWraper(kmeans_data=self.data, wraper_output=self.WraperOutput, model_name=self.model_name, base_path=self.kwargs.get(BASE_PATH_DEST))
        self.IOHandler.create_output_dirs()
        self.IOHandler.merge_kmeans_data()
        self.IOHandler.move_kmeans_data()
        logging.info(f"Check {OUTPUT_DIR_PATH} for output")

        inception_data = self._loader.create_inception_data()
        self.WraperOutput = self.inception.run()
        self.IOHandler.set_inception_data(inception_data=inception_data, wraper_output=self.WraperOutput)

        self.IOHandler.merge_inception_data(self.kwargs.get(TOP_DEST))
        self.IOHandler.move_inception_data()
