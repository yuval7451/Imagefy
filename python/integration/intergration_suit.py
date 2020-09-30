#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
from integration.kmeans_tensorflow_wraper import KmeansTensorflowWraper
from integration.data_utils import DataLoaderWraper, IOWraper
from integration.visualization import to_visualization_format, ComputePCA, Visualize3D
from integration.common import OUTPUT_DIR_PATH, VISUALIZATION_DIM

class IntergrationSuit:
    """
    """
    def __init__(self, dir_path, start_k, stop_k, image_size):
        """
        """
        # Asign Parameters
        self._dir_path = dir_path
        self._start_k = start_k
        self.stop_k = stop_k
        self.image_size = image_size
        self.data = None
        self.score = None
        self.IOHandler = None

    def run(self):
        """
        """
        # Load Data
        self.data = DataLoaderWraper(self._dir_path, self.image_size).run()
        # Initialize Kmeans and run it
        cluster = KmeansTensorflowWraper(self.data, self._start_k, self.stop_k)
        self.score = cluster.run()
        # Handle IO And File Transfering
        self.IOHandler = IOWraper(data=self.data, score=self.score)
        self.IOHandler.create_output_dirs()
        self.IOHandler.merge_data()
        self.IOHandler.move_data()
        logging.info(f"Check {OUTPUT_DIR_PATH} for output")

    def visualize(self):
        """
        """
        if self.data is not None:
            x, y = to_visualization_format(data=self.data)
            pca = ComputePCA(x, VISUALIZATION_DIM)
            X = pca.run()
            vis = Visualize3D(data=X, labels=y)
            vis.show()
        else:
            raise RuntimeError("Cannon visualize befor IntergrationSuit.run() was Invoked or use IntergrationSuit.visualize_from_file")

    def visualize_from_file(self, filename):
        vis = Visualize3D(data=None, labels=None)
        vis.load(filename)
        vis.show()