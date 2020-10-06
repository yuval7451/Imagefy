#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import os
import logging
import datetime
from abc import ABC, abstractclassmethod
from integration.wrapers.mini_batch_kmeans_tensorflow_wraper import MiniBatchKmeansTensorflowWraper
from integration.utils.common import BASE_PATH_DEST, LOG_DIR, MINI_KMEAND_DEST, WRAPER_PARAM

class BaseSuit(ABC):
    """BaseSuit -> Some Kind of Class that controls everything."""
    def __init__(self, **kwargs):
        """
        """
        logging.debug(f"Initializing {self.__class__.__name__}")
        self.kwargs = kwargs
        # Asign Paramete
        self._wrapers = self._get_wrapers()
        assert self.kwargs.get(WRAPER_PARAM) != None
        self._wraper = None
        self._loader = None
        self.WraperOutput = None
        self.IOHandler = None
        (self.model_name, self.base_path, self.base_model_dir) = self._set_model_directories()

    @abstractclassmethod
    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseWraper & @IOWraper
        """
        logging.info(f"Starting {self.__class__.__name__}.run()")

    def _get_wrapers(self):
        _wrapers = {
            MINI_KMEAND_DEST: MiniBatchKmeansTensorflowWraper,
        }
        logging.debug(f"Loading {len(_wrapers)} Wrapers")
        return _wrapers

    def _set_model_directories(self):
        wraper_name = self._wrapers.get(self.kwargs.get(WRAPER_PARAM)).__name__
        base_path = self.kwargs.get(BASE_PATH_DEST)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        model_name = f"{wraper_name}-{current_time}" 
        base_path = base_path
        base_model_dir = os.path.join(base_path, LOG_DIR, model_name)
        return (model_name, base_path, base_model_dir)