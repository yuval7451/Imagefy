#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
from abc import ABC, abstractclassmethod
from imagefy_clustering.wrapers.mini_batch_kmeans_tensorflow_wraper import MiniBatchKmeansTensorflowWraper
from imagefy_clustering.wrapers.kmeans_tensorflow_wraper import KmeansTensorflowWraper 
from imagefy_clustering.utils.data_utils import DataLoader, TensorLoader
from imagefy_clustering.utils.common import KMEANS_DEST, MINI_KMEAND_DEST, DATA_LOADER_DEST, TENSOR_LADER_DEST, WRAPER_PARAM

class BaseSuit(ABC):
    """BaseSuit -> Some Kind of Class that controls everything."""
    def __init__(self, **kwargs):
        """
        """
        logging.debug(f"Initializing {self.__class__.__name__}")
        self.kwargs = kwargs
        # Asign Paramete

        self._wrapers = self._get_wrapers()
        if self.kwargs.get(WRAPER_PARAM) is None:
            raise RuntimeError("wraper is missing, use <kmeans, mini_kmeans>")
        self._loaders = self._get_loaders()
        self._wraper = None
        self._loader = None
        self.WraperOutput = None
        self.IOHandler = None

    @abstractclassmethod
    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseWraper & @IOWraper
        """
        logging.info(f"Starting {self.__class__.__name__}.run()")

    def _get_wrapers(self):
        _wrapers = {
            MINI_KMEAND_DEST: MiniBatchKmeansTensorflowWraper,
            KMEANS_DEST: KmeansTensorflowWraper,
        }
        logging.debug(f"Loading {len(_wrapers)} Wrapers")
        return _wrapers
        # _wrapers_path = self._get_wrapers_path()
        # module = __import__(module_name)
        # my_class = getattr(module, class_name)
        # instance = my_class()

    def _get_loaders(self):
        _loaders = {
            TENSOR_LADER_DEST: TensorLoader,
            DATA_LOADER_DEST: DataLoader,
        }
        logging.debug(f"Loading {len(_loaders)} Loaders")
        return _loaders

    # def _get_wrapers_path(self):
    #     _wrapers_path = [os.path.join(WRAPERS_DIR, wraper).replace("\\", ".") for wraper in os.listdir(WRAPERS_DIR) if wraper.endswith(WRAPER_PREFIX)]
    #     logging.info(f"Found {len(_wrapers_path)} Wrapers")
    #     logging.debug(str(_wrapers_path))
    #     return _wrapers_path
