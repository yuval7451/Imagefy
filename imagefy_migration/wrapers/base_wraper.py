#! /usr/bin/env python3
"""
Author: Yuval Kaneti
"""
#### IMPORTS ####
import logging
import numpy as np
from imagefy_migration.utils.common import MINI_BATCH_KMEANS_TENSORFLOW_WRAPER, INCEPTION_RESNET_TENSORFLOW_WRAPER
from imagefy_migration.utils.data_utils import BaseLoader
from imagefy_migration.plugins.tensorboard import TensorboardWraper
from abc import ABC, abstractmethod

class BaseWraper(ABC):
    """BaseWraper -> An Abstract Class for TensorflowWrapers."""
    def __init__(self, data: list, tensorboard: bool, loader: BaseLoader, base_model_dir: str, **kwrags):
        """
        @param data: C{list} -> A list of `hollow` Image object used for IOWraper.
        @param tensrboard :C{bool} -> Whether to log tensorboard files or not.
        @pram loader: C{BaseLoader} -> The Data Loader Allocated for the Wraper.
        @param base_model_dir: C{str} -> The Base model dir for output & logs, taken for BaseSuit.base_model_dir.
        @param kwargs: C{dict} -> For future Resons.
        @local name: C{str} -> the name of the class, including parents.
        @local _input_functions: C{list} -> A list of all suported input functions provided bu Loaders.
        @local _input_fn: C{callable} -> The input function The Wraper will use, Chosen by his name.
        @local wraper_output: C{BaseWraper} -> A placeholder for the wraper output.
        """
        self._data = data
        self._use_tensorboard = tensorboard
        self.base_model_dir = base_model_dir
        self._kwrags = kwrags
        self._loader = loader
        self.name = self.__class__.__name__
        self._input_functions = {
            MINI_BATCH_KMEANS_TENSORFLOW_WRAPER: self._loader.mini_batch_kmeans_input_fn,
            INCEPTION_RESNET_TENSORFLOW_WRAPER: self._loader.inception_input_fn,
        }
        self._input_fn = self._input_functions.get(self.name)
        self.wraper_output = None
        logging.debug(f"Initilazing {self.name}")

    @abstractmethod
    def run(self):
        """
        @remarks *abstract method run.
        """
        logging.info(f"Starting {self.name}")

    def Wraperize(self):
        """
        @return C{tuple} -> [Image.data], [Image.src_path]
        @remarks *..
        """
        X = np.asarray([ImageObj.data for ImageObj in self._data])
        filenames = [ImageObj.src_path for ImageObj in self._data]
        return (X, filenames)

    def update(self):
        """
        @remarks *updates the Image.cluster_n to match The cluster label, ussely done with IOWraper.merge_data(...).
        """
        for (image, cluster_label) in zip(self._data, self.wraper_output.cluster_labels):
            image.cluster_n = int(cluster_label)

    def _tensorboard(self, batch_size: int):
        """
        @param batch_size: C{int} -> used for the Tensorboard Dataset Loader.
        @remarks *sometimes labels & filenames dont match, they will be truncated to the shortest one. 
        """
        labels = self.wraper_output.cluster_labels
        filenames = self._loader._image_names
        if len(labels) != len(filenames):
            logging.debug(f"Labels from Data Loader & filenames from IOWraper dont match\n fileanems will be truncated, this is a known issue.")

        tensor_board = TensorboardWraper(name=self.name,
                                        base_model_dir=self.base_model_dir,
                                        metadata=(labels, filenames), 
                                        batch_size=batch_size, 
                                        data_length=len(labels), 
                                        dataset=self._input_fn,
                                        **self._kwrags)
        tensor_board.save_labels()
        tensor_board.run()
