#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import os
import logging
import datetime
from abc import ABC, abstractclassmethod
from imagefy.utils.common import BASE_PATH_DEST, LOG_DIR, WRAPER_PARAM, OUTPUT_DIR_PATH

class BaseSuit(ABC):
    """BaseSuit -> Some Kind of Class that controls everything."""
    def __init__(self, **kwargs: dict):
        """
        @param kwargs: C{dict} -> A dict with all parameters passed on Runtime.
        @remarks *Base Class for Suits.
        """
        self.name = self.__class__.__name__
        logging.debug(f"Initializing {self.name}")
        self.kwargs = kwargs
        # Asign Paramete
        self._loader = None
        self.WraperOutput = None
        self.IOHandler = None
        (self.model_name, self.base_path, self.base_model_dir, self.output_dir_path) = self._set_model_directories()

    @abstractclassmethod
    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseWraper & @IOWraper
        """
        logging.info(f"Starting {self.name}")

    def _set_model_directories(self):
        """
        """
        base_path = self.kwargs.get(BASE_PATH_DEST)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        model_name = f"{self.name}-{current_time}" 
        base_path = base_path
        base_model_dir = os.path.join(base_path, LOG_DIR, model_name)
        output_dir_path = os.path.join(base_path, OUTPUT_DIR_PATH, model_name, "*", "*")
        return (model_name, base_path, base_model_dir, output_dir_path)