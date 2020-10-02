#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
from integration.wrapers.kmeans_tensorflow_wraper import KmeansTensorflowWraper
from integration.wrapers.mini_batch_kmeans_tensorflow_wraper import MiniBatchKmeansTensorflowWraper
from integration.utils.data_utils import DataLoaderWraper, IOWraper, TensorboardWraper
from integration.utils.common import OUTPUT_DIR_PATH, VISUALIZATION_DIM
from integration.suits.base_suit import BaseSuit
class IntergrationSuit(BaseSuit):
    """IntergrationSuit -> Some Kind of Class that controls everything."""
    def __init__(self, **args):
        """
        dir_path : str, start_k: int, stop_k: int, image_size: int
        @param dir_path: C{str} -> The base dir path
        @param start_k: C{int} -> used for KmeansTensorflowWraper.
        @param stop_k: C{int} -> used for KmeansTensorflowWraper.
        @param image_size: C{int} -> the image size the images will be resized to. (DataLoaderWraper)
        @rermarks *Other the dir_path & image_size all should pass ar *args & **kwargs
                  *BaseSuit ?
        """
        logging.debug("Initializing IntergrationSuit")
        self.args = args
        print(self.args)
        # Asign Parameters
        self.WraperOutput = None
        self.IOHandler = None

    def run(self):
        """
        The `main` Function of each Suit, usually calls The @BaseWraper & @IOWraper
        """
        logging.info("Starting IntergrationSuit.run()")
        # Load Data
        self.data = DataLoaderWraper(self._dir_path, self.image_size).run()
        # Initialize Kmeans and run it
        cluster = MiniBatchKmeansTensorflowWraper(data=self.data)
        cluster.run()
        # cluster.save()

        self.WraperOutput = cluster.transform()
        # cluster = KmeansTensorflowWraper(self.data, self._start_k, self.stop_k)
        # self.WraperOutput = cluster.run()
        # # Handle IO And File Transfering
        self.IOHandler = IOWraper(data=self.data, score=self.WraperOutput, model_name=cluster.model_name)
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