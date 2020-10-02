#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
from abc import ABC, abstractclassmethod
from integration.utils.common import WRAPERS_DIR, WRAPER_PREFIX
from integration.wrapers.mini_batch_kmeans_tensorflow_wraper import MiniBatchKmeansTensorflowWraper, MiniBatchKmeansWraperOutput
from integration.wrapers.kmeans_tensorflow_wraper import KmeansTensorflowWraper, KmeansWraperOutput 
from integration.utils.common import KMEANS_DEST, MINI_KMEAND_DEST 

class BaseSuit(ABC):
    """BaseSuit -> Some Kind of Class that controls everything."""
    def __init__(self, **kwargs):
        """
        """
        logging.debug(f"Initializing {self.__class__.__name__}")
        self.kwargs = kwargs
        # Asign Paramete
        self._wrapers = self._get_wrapers()
        self._wraper = None
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

    # def _get_wrapers_path(self):
    #     _wrapers_path = [os.path.join(WRAPERS_DIR, wraper).replace("\\", ".") for wraper in os.listdir(WRAPERS_DIR) if wraper.endswith(WRAPER_PREFIX)]
    #     logging.info(f"Found {len(_wrapers_path)} Wrapers")
    #     logging.debug(str(_wrapers_path))
    #     return _wrapers_path
