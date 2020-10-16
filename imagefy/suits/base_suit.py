"""
@Author: Yuval Kaneti 
"""


#### Imports ####
import os
import logging; logger = logging.getLogger('Imagefy')
import datetime
import tensorflow as tf
from abc import ABC, abstractclassmethod
from imagefy.www.common import PROGRESS_10
from imagefy.utils.common import BASE_PATH_PARAM, LOG_DIR, LOG_FILENAME, MODEL_NAME_PARAM,\
    BASE_PATH_PARAM, BASE_MODEL_DIR_PARAM

class BaseSuit(ABC):
    """BaseSuit -> Some Kind of Class that controls everything."""
    def __init__(self, progress_bar: object, **kwargs: dict):
        """
        @param kwargs: C{dict} -> A dict with all parameters passed on Runtime.
        @remarks *Base Class for Suits.
        """
        self.name = self.__class__.__name__
        logger.debug(f"Initializing {self.name}")
        self.progress_bar = progress_bar
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
        logger.info(f"Starting {self.name}")

    def _set_model_directories(self):
        """
        @remarks *Sets the model base dir & name.
        """
        base_path = self.kwargs.get(BASE_PATH_PARAM)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d")
        model_name = f"{self.name}-{current_time}" 
        base_path = base_path
        base_model_dir = os.path.join(base_path, LOG_DIR, model_name)
        if not os.path.exists(base_model_dir):
            os.makedirs(base_model_dir)

        return (model_name, base_path, base_model_dir)

    def Initialize(self):
        # importent directories for the model
        (self.model_name, 
        self.base_path, 
        self.base_model_dir) = self._set_model_directories()
        # logger & stuff
        self.initialize_logger()
        # Gpu's
        self.initialize_gpu()
        # Kwargs
        self.initialize_kwargs()

        self.progress_bar.progress(PROGRESS_10)

    def initialize_logger(self):
        # Log files
        logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.ERROR)

        log_path = os.path.join(self.base_model_dir, LOG_FILENAME)        
        FileHandler = logging.FileHandler(log_path)
        FileHandler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(levelname)s - %(name)s - %(filename)s - %(funcName)s - %(asctime)s - %(message)s')
        FileHandler.setFormatter(formatter)

        tensorflow_level = tf.compat.v1.logging.INFO
        logging.getLogger('tensorflow').addHandler(FileHandler)
        tf.compat.v1.logging.set_verbosity(tensorflow_level)

        logger.addHandler(FileHandler)
        logger.setLevel(logging.INFO)

    def initialize_gpu(self):
        gpu_avilable = len(tf.config.experimental.list_physical_devices('GPU'))
        logger.info(f"Num GPUs Available: {gpu_avilable}") 
        gpu_log_level = False
        tf.debugging.set_log_device_placement(gpu_log_level)
        logger.info(f"logger GPU device placement: {gpu_log_level}")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        tf.config.set_soft_device_placement(True)
        logger.info(f"Tensorflow is Executing Eagerly: {tf.executing_eagerly()}")

    def initialize_kwargs(self):
        self.kwargs.update({
            MODEL_NAME_PARAM: self.model_name, 
            BASE_PATH_PARAM: self.base_path, 
            BASE_MODEL_DIR_PARAM: self.base_model_dir})

        logger.debug(str(self.kwargs))
