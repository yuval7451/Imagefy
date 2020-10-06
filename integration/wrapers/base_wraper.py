#! /usr/bin/env python3
"""
Author: Yuval Kaneti⭐
"""
#### IMPORTS ####
import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import logging
import numpy as np
import tensorflow as tf; tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from integration.utils.common import LOG_DIR, MINI_BATCH_KMEANS_TENSORFLOW_WRAPER, INCEPTION_RESNET_TENSORFLOW_WRAPER
from integration.utils.data_utils import BaseLoader
from integration.plugins.tensorboard import TensorboardWraper
from abc import ABC, abstractmethod

class BaseWraper(ABC):
    """BaseWraper -> An Abstract Class for TensorflowWrapers."""
    def __init__(self, data: list, tensorboard: bool, loader: BaseLoader, base_model_dir: str, **kwrags):
        self._data = data
        self._use_tensorboard = tensorboard
        self.name = self.__class__.__name__
        self.base_model_dir = base_model_dir
        self._kwrags = kwrags
        self._loader = loader
        self._input_functions = {
            MINI_BATCH_KMEANS_TENSORFLOW_WRAPER: self._loader.mini_batch_kmeans_input_fn,
            INCEPTION_RESNET_TENSORFLOW_WRAPER: self._loader.inception_input_fn,
        }
        self._input_fn = self._input_functions.get(self.name)
        self.wraper_output = None
        logging.debug(f"Initilazing {self.name}")

    @abstractmethod
    def run(self):
        logging.info(f"Starting {self.name}")

    def Wraperize(self):
        X = np.asarray([ImageObj.data for ImageObj in self._data])
        filenames = [ImageObj.src_path for ImageObj in self._data]
        logging.debug(f"Wraperize X: {X.shape}")
        return (X, filenames)

    def update(self):
        for (image, cluster_label) in zip(self._data, self.wraper_output.cluster_labels):
            image.cluster_n = int(cluster_label)

    def _tensorboard(self, batch_size: int):
        labels = self.wraper_output.cluster_labels
        filenames = self._loader._image_names
        if len(labels) != len(filenames):
            logging.debug(f"Labels (Cluster output): {len(labels)}, filenames (os.listdir()): {len(filenames)}")

        tensor_board = TensorboardWraper(name=self.name,
                                        base_model_dir=self.base_model_dir,
                                        metadata=(labels, filenames), 
                                        batch_size=batch_size, 
                                        data_length=len(labels), 
                                        dataset=self._input_fn,
                                        **self._kwrags)
        tensor_board.save_labels()
        tensor_board.run()
