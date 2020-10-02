#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
# from integration.wrapers.kmeans_tensorflow_wraper import KmeansTensorflowWraper
# from integration.wrapers.mini_batch_kmeans_tensorflow_wraper import MiniBatchKmeansTensorflowWraper
from integration.suits.base_suit import BaseSuit
from integration.utils.data_utils import DataLoaderWraper, IOWraper, TensorboardWraper
from integration.utils.common import DIR_DEST, IMAGE_SIZE, OUTPUT_DIR_PATH, TENSORBOARD_DEST, DATA_PARAM, VERBOSE_DEST, WRAPER_PARAM, TENSORBOARD_NAME_DEST, SIZE_DEST

class IntergrationSuit(BaseSuit):
    """IntergrationSuit -> Some Kind of Class that controls everything."""
    def __init__(self, **kwargs):
        """
        """
        super().__init__(**kwargs)
        # Asign Parameters
        self.kwargs = kwargs
        self.verbose = self.kwargs[VERBOSE_DEST]
        logging.getLogger().setLevel(logging.DEBUG if self.verbose else logging.INFO)
        self.WraperOutput = None
        self.IOHandler = None
        self.tensoboard = self.kwargs[TENSORBOARD_DEST]
        self.tensorboard_name = self.kwargs[TENSORBOARD_NAME_DEST]
        logging.debug(str(self.kwargs)  )

    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseWraper & @IOWraper
        """
        # Load Data
        self.data = DataLoaderWraper(self.kwargs[DIR_DEST], self.kwargs[SIZE_DEST]).run()
        self.kwargs.update({DATA_PARAM: self.data})
        self._wraper = self._wrapers[self.kwargs[WRAPER_PARAM]](**self.kwargs)
        self.WraperOutput = self._wraper.run()
        
        # Handle IO And File Transfering
        self.IOHandler = IOWraper(data=self.data, wraper_output=self.WraperOutput, model_name=self._wraper.model_name)
        self.IOHandler.create_output_dirs()
        self.IOHandler.merge_data()
        self.IOHandler.move_data()
        logging.info(f"Check {OUTPUT_DIR_PATH} for output")
        if self.tensoboard:
            self.tensorboard()

    def tensorboard(self, filename : str=None):
        """
        """
        if self.data is None and filename is not None:
            tensor_board = TensorboardWraper(data=None)
            tensor_board.load(filename=filename)
            tensor_board.create_tensorboard_output("Imagefy")
        else:
            tensor_board = TensorboardWraper(data=self.data)
            tensor_board.save()
            tensor_board.create_tensorboard_output("Imagefy")