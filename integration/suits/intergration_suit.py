#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
from integration.wrapers.kmeans_tensorflow_wraper import KmeansTensorflowWraper
from integration.utils.data_utils import DataLoaderWraper, IOWraper, DataLoader, TensorboardWraper
from integration.utils.common import OUTPUT_DIR_PATH, VISUALIZATION_DIM

class IntergrationSuit:
    """IntergrationSuit -> Some Kind of Class that controls everything."""
    def __init__(self, dir_path : str, start_k: int, stop_k: int, image_size: int):
        """
        @param dir_path: C{str} -> The base dir path
        @param start_k: C{int} -> used for KmeansTensorflowWraper.
        @param stop_k: C{int} -> used for KmeansTensorflowWraper.
        @param image_size: C{int} -> the image size the images will be resized to. (DataLoaderWraper)
        @rermarks *Other the dir_path & image_size all should pass ar *args & **kwargs
                  *BaseSuit ?
        """
        logging.debug("Initializing IntergrationSuit")
        # Asign Parameters
        self._dir_path = dir_path
        self._start_k = start_k
        self.stop_k = stop_k
        self.image_size = image_size
        self.data = None
        self.score = None
        self.IOHandler = None

    async def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseWraper & @IOWraper
        """
        logging.info("Starting @IntergrationSuit.run()")
        # Load Data
        self.data = DataLoaderWraper(self._dir_path, self.image_size).run()
        # self.data = await DataLoader(self._dir_path, self.image_size).run()
        # Initialize Kmeans and run it
        cluster = KmeansTensorflowWraper(self.data, self._start_k, self.stop_k)
        self.score = cluster.run()
        # # Handle IO And File Transfering
        self.IOHandler = IOWraper(data=self.data, score=self.score)
        self.IOHandler.create_output_dirs()
        self.IOHandler.merge_data()
        self.IOHandler.move_data()
        logging.info(f"Check {OUTPUT_DIR_PATH} for output")

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