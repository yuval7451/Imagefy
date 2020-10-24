# Author: Yuval Kaneti

## Imports
import os
import shutil; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging 
from glob import glob
import tensorflow as tf
from imagefy.suits.base_suit import BaseSuit
from imagefy.utils.data_utils import IOWraper, TensorLoader
from imagefy.utils.common import  BASE_PATH_DEST, DATA_PARAM, DIR_DEST, TOP_DEST, LOADER_DEST, FILENAMES_DEST
from imagefy.wrapers.mini_batch_kmeans_tensorflow_wraper import MiniBatchKmeansTensorflowWraper
from imagefy.wrapers.inception_resnet_tensorflow_wraper import InceptionResnetTensorflowWraper

class IntergrationSuit(BaseSuit):
    """IntergrationSuit -> Some Kind of Class that controls everything."""
    def __init__(self, kmeans_batch_size: int, **kwargs: dict):
        """
        @param kwargs: C{dict} -> A dict with all parameters passed on Runtime.
        @remarks:
                 *Curently the boss of the module.
        """
        super().__init__(**kwargs)
        self.kmeans_batch_size = kmeans_batch_size

    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseLoader, @BaseWraper & @IOWraper
        """
        # Load Data
        self._loader = TensorLoader(**self.kwargs)
        filenames = glob(self.kwargs.get(DIR_DEST)) # type: ignore
        for slice in range(0, len(filenames), self.kmeans_batch_size):
            if slice + self.kmeans_batch_size > len(filenames):
                filenames_batch = filenames[:slice]
            else:    
                filenames_batch = filenames[slice:slice+ self.kmeans_batch_size]
        
            self._loader._image_names = filenames_batch
            self.data = self._loader.run()
            self.kwargs.update({DATA_PARAM: self.data, LOADER_DEST: self._loader, FILENAMES_DEST: filenames_batch}) # type: ignore
        
            # Kmeans
            self.kmeans = MiniBatchKmeansTensorflowWraper(**self.kwargs) 
            self.WraperOutput = self.kmeans.run()
            
            # Handle IO And File Transfering
            self.IOHandler = IOWraper(kmeans_data=self.data, wraper_output=self.WraperOutput, model_name=self.model_name, base_path=self.kwargs.get(BASE_PATH_DEST)) # type: ignore
            self.IOHandler.create_output_dirs()
            self.IOHandler.merge_kmeans_data()
            self.IOHandler.move_kmeans_data()
            logging.info(f"Check {self.base_model_dir} for output")
            if os.path.exists(self.base_model_dir):
                shutil.rmtree(self.base_model_dir)
                os.makedirs(self.base_model_dir)
            else:
                os.makedirs(self.base_model_dir)
        # Inception
        self.inception = InceptionResnetTensorflowWraper(**self.kwargs)
        inception_data = self._loader.create_inception_data()
        self.WraperOutput = self.inception.run()
        self.IOHandler.set_inception_data(inception_data=inception_data, wraper_output=self.WraperOutput)

        self.IOHandler.merge_inception_data(self.kwargs.get(TOP_DEST)) # type: ignore
        self.IOHandler.move_inception_data()
        self.IOHandler.save_image_predictions()

        logging.info("Finished running Suit")
