#! /usr/bin/env python3
"""
Author: Yuval Kaneti⭐
"""
#### IMPORTS ####
import os
import time
import logging
import numpy as np
from integration.utils.common import TENSORBOARD_LOG_DIR
from abc import ABC, abstractmethod

class BaseWraper(ABC):
    """BaseWraper -> An Abstract Class for TensorflowWrapers."""
    def __init__(self):
        self._data = None
        self.name = self.__class__.__name__
        self.model_name = f"{self.name}-{time.time()}" 
        self.model_dir=os.path.join(os.getcwd(),TENSORBOARD_LOG_DIR, self.model_name)
    
    @abstractmethod
    def run(self):
        logging.info(f"Starting {self.__class__.__name__}")


    def Wraperize(self):
        X = np.asarray([ImageObj.data for ImageObj in self._data])
        y = [ImageObj.src_path for ImageObj in self._data]
        return (X, y)