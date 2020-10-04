#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
from integration.suits.base_suit import BaseSuit
from integration.utils.data_utils import IOWraper
from integration.utils.common import  BASE_PATH_DEST, OUTPUT_DIR_PATH, DATA_PARAM, VERBOSE_DEST, WRAPER_PARAM, LOADER_DEST

class IntergrationSuit(BaseSuit):
    """IntergrationSuit -> Some Kind of Class that controls everything."""
    def __init__(self, **kwargs):
        """
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
        logging.debug(str(self.kwargs))

    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseWraper & @IOWraper
        """
        # Load Data
        loader_name = self.kwargs.get(LOADER_DEST)
        self._loader = self._loaders.get(loader_name)(**self.kwargs)   #self._loaders[self.kwargs[LOADER_DEST]](**self.kwargs)
        self.data = self._loader.hollow_images() if self._loader.dtype is tf.data.Dataset else self._loader.run()
        self.kwargs.update({DATA_PARAM: self.data, LOADER_DEST: self._loader})
       
        wraper_name = self.kwargs.get(WRAPER_PARAM)
        self._wraper = self._wrapers.get(wraper_name)(**self.kwargs)   #self._wrapers[self.kwargs[WRAPER_PARAM]](**self.kwargs)
        self.WraperOutput = self._wraper.run()
        
        # Handle IO And File Transfering
        self.IOHandler = IOWraper(data=self.data, wraper_output=self.WraperOutput, model_name=self._wraper.model_name, base_path=self.kwargs.get(BASE_PATH_DEST))
        self.IOHandler.create_output_dirs()
        self.IOHandler.merge_data()
        self.IOHandler.move_data()
        logging.info(f"Check {OUTPUT_DIR_PATH} for output")
