#!/usr/bin/env python3
# Author: Yuval Kaneti

#### Imports ####
import os
import logging 
import datetime
import tensorflow as tf
from abc import ABC, abstractclassmethod
from imagefy.utils.common import BASE_PATH_DEST, LOG_DIR, LOG_FILENAME, OUTPUT_DIR_PATH, MODEL_NAME_PARAM,\
    BASE_PATH_DEST, BASE_MODEL_DIR_PARAM, OUTPUT_DIR_PATH_PARAM

class BaseSuit(ABC):
    """BaseSuit -> Some Kind of Class that controls everything."""
    def __init__(self, verbose: bool, **kwargs: dict):
        """
        @param kwargs: C{dict} -> A dict with all parameters passed on Runtime.
        @remarks *Base Class for Suits.
        """
        self.name = self.__class__.__name__
        logging.debug(f"Initializing {self.name}")
        self.verbose = verbose

        self.kwargs = kwargs
        self._loader = None
        self.WraperOutput = None
        self.IOHandler = None
        self.Initialize()
        
    @abstractclassmethod
    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseWraper & @IOWraper
        """
        logging.info(f"Starting {self.name}")

    def _set_model_directories(self):
        """
        @remarks *Sets the model base dir & name.
        """
        base_path = self.kwargs.get(BASE_PATH_DEST)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        model_name = f"{self.name}-{current_time}" 
        base_path = base_path
        base_model_dir = os.path.join(base_path, LOG_DIR, model_name)
        output_dir_path = os.path.join(base_path, OUTPUT_DIR_PATH, model_name, "*", "*")
        os.makedirs(base_model_dir)
        return (model_name, base_path, base_model_dir, output_dir_path)

    def Initialize(self):
        # importent directories for the model
        (self.model_name, 
        self.base_path, 
        self.base_model_dir, 
        self.output_dir_path) = self._set_model_directories()
        # Logging & stuff
        self.initialize_logging()
        # Gpu's
        self.initialize_gpu()
        # Kwargs
        self.initialize_kwargs()

    def initialize_logging(self):
        # Log files
        #FIX ME:
        log_path = os.path.join(self.base_model_dir, LOG_FILENAME)        
        FileHandler = logging.FileHandler(log_path)
        FileHandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(filename)s - %(funcName)s - %(asctime)s - %(message)s')
        FileHandler.setFormatter(formatter)
        
        level = logging.DEBUG if self.verbose else logging.INFO
        tensorflow_level = tf.compat.v1.logging.INFO if self.verbose else tf.compat.v1.logging.ERROR
        logging.getLogger('tensorflow').addHandler(FileHandler)
        tf.compat.v1.logging.set_verbosity(tensorflow_level)

        logging.getLogger().addHandler(FileHandler)
        logging.getLogger().setLevel(level)



    def initialize_gpu(self):
        gpu_avilable = len(tf.config.experimental.list_physical_devices('GPU'))
        logging.info(f"Num GPUs Available: {gpu_avilable}") 
        gpu_log_level = False #True if self.verbose else False
        tf.debugging.set_log_device_placement(gpu_log_level)
        logging.info(f"Logging GPU device placement: {gpu_log_level}")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        tf.config.set_soft_device_placement(True)
        logging.info(f"Tensorflow is Executing Eagerly: {tf.executing_eagerly()}")
        # tf.profiler.experimental.server.start(6009)

    def initialize_kwargs(self):
        self.kwargs.update({
            MODEL_NAME_PARAM: self.model_name, 
            BASE_PATH_DEST: self.base_path, 
            BASE_MODEL_DIR_PARAM: self.base_model_dir,
            OUTPUT_DIR_PATH_PARAM: self.output_dir_path})

        logging.debug(str(self.kwargs))
