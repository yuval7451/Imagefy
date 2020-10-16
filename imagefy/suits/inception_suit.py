"""
@Author: Yuval Kaneti 
"""


#### Imports ####
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging; logger = logging.getLogger('Imagefy')
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

from imagefy.suits.base_suit import BaseSuit
from imagefy.utils.data_utils import IOWraper, TensorLoader
from imagefy.www.common import PROGRESS_25, PROGRESS_50, PROGRESS_75
from imagefy.utils.common import  DATA_PARAM, IMAGES_PARAM, TOP_PARAM, LOADER_PARAM
from imagefy.wrapers.inception_resnet_tensorflow_wraper import InceptionResnetTensorflowWraper

class InceptionSuit(BaseSuit):
    """InceptionSuit -> Some Kind of Class that controls everything."""
    def __init__(self, **kwargs: dict):
        """
        @param kwargs: C{dict} -> A dict with all parameters passed on Runtime.
        @remarks *Curently the bos of the module.
        """
        super().__init__(**kwargs)


    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseLoader, @BaseWraper & @IOWraper
        """
        # Load Data
        self._loader = TensorLoader(**self.kwargs)  
        self.images = self._loader.run()
        self.kwargs.update({DATA_PARAM: self.images, LOADER_PARAM: self._loader})
        self.progress_bar.progress(PROGRESS_25)


        # Handle IO And File Transfering
        self.IOHandler = IOWraper(images=self.images, wraper_output=None, model_name=self.model_name, base_path=self.base_path)
        self.IOHandler.reset_cluster_labels()

        self.kwargs.update({IMAGES_PARAM: self.images})
        # Inception
        self.inception = InceptionResnetTensorflowWraper(**self.kwargs)
        self.progress_bar.progress(PROGRESS_50)

        self.WraperOutput = self.inception.run()
        self.progress_bar.progress(PROGRESS_75)

        self.IOHandler.set_inception_data( wraper_output=self.WraperOutput)

        self.images = self.IOHandler.merge_inception_data(self.kwargs.get(TOP_PARAM))


        logger.info("Finished running Suit")
        return self.images
