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
from imagefy_clustering.utils.common import TENSORBOARD_LOG_DIR, LOG_DIR, MODEL_LOG_DIR
from imagefy_clustering.utils.data_utils import BaseLoader
from imagefy_clustering.plugins.tensorboard import TensorboardWraper, Tensorboard
from abc import ABC, abstractmethod

class BaseWraper(ABC):
    """BaseWraper -> An Abstract Class for TensorflowWrapers."""
    def __init__(self, data: list, tensorboard: bool, loader: BaseLoader, base_path: str, **kwrags):
        self._data = data
        self._use_tensorboard = tensorboard
        self.name = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.model_name = f"{self.name}-{current_time}" 
        self.base_path = base_path
        self.base_model_dir = os.path.join(self.base_path, LOG_DIR, self.model_name)
        self.model_dir = os.path.join(self.base_model_dir, MODEL_LOG_DIR)
        self.tensorboard_dir = os.path.join(self.base_model_dir, TENSORBOARD_LOG_DIR)
        logging.basicConfig(filename=f"{os.path.join(self.base_model_dir, 'session.log')}", filemode='w')
        self._kwrags = kwrags
        self._loader = loader
        self._input_fn = self._loader.run if self._loader.dtype is tf.data.Dataset else self.input_fn
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

    def input_fn(self, num_epochs: int, **kwrags):
        (self.X, filenames) = self.Wraperize()
        # For Futere Refrence
        return tf.data.Dataset.from_tensors(self.X).repeat(num_epochs)
        # return tf.compat.v1.train.limit_epochs(
        #     tf.convert_to_tensor(self.X, dtype=tf.float32), num_epochs=num_epochs)

    def update(self):
        for (image, cluster_label) in zip(self._data, self.wraper_output.cluster_labels):
            image.cluster_n = int(cluster_label)

    def _tensorboard(self, dtype: type, batch_size: int):
        if dtype is list:
            X, filenames = self.Wraperize()
            y = self.wraper_output.cluster_labels
            tensor_board = Tensorboard(data=None, X=X, y=(y, filenames), name=self.name, base_model_dir=self.base_model_dir, **self._kwrags)
            tensor_board.run()  
        else:
            y = self.wraper_output.cluster_labels
            filenames = self._loader._image_names
            # assert len(y) == len(filenames)
            if len(y) != len(filenames):
                logging.debug(f"Labels (Cluster output): {len(y)}, filenames (os.listdir()): {len(filenames)}")
                # logging.debug(y)
                # logging.debug(filenames)
                logging.warn("Error might be coming")
                # filenames = filenames[:len(y)]

            tensor_board = TensorboardWraper(name=self.name,
                                             base_model_dir=self.base_model_dir,
                                             y=(y, filenames), 
                                             batch_size=batch_size, 
                                             data_length=len(y), 
                                             **self._kwrags)
            tensor_board.save_labels()
            tensor_board.run()
    
        # @tf.function
