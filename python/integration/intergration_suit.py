#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
from integration.kmeans_tensorflow_wraper import KmeansTensorflowWraper
from integration.data_utils import DataLoaderWraper, IOWraper
from integration.visualization import to_visualization_format, ComputePCA, Visualize3D
from integration.common import OUTPUT_DIR_PATH, VISUALIZATION_DIM

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
        The `main` Function of each Suit, usually calls The @BaseWraper & @IOWraper
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
        @remarks *Will Compute PCA on the data to transfor it to 3-dim from N-dim.
                 *Will then plot it on a 3d Grid via matplotlib.
                 *You Can use visualize_from_file if you already saved the PCA'd data, (Happends automatically).
                 *Might take awhile to finish...

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
        """
        @remarks *Loads .npy Data saved using PCA and shows it.
        """
        vis = Visualize3D(data=None, labels=None)
        vis.load(filename)
        vis.show()