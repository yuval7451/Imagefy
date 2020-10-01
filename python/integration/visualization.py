#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import time
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from integration.common import VISUALIZATION_VECTOR_SIZE, COLORS

class ComputePCA():
    """ComputePCA -> An implemntion of PCA in Tensorflow."""
    def __init__(self, data : np.ndarray, dim : int):
        """
        @param data: C{np.ndarray} -> An np.ndarray of (Image.data).
        @param dim: C{int} -> the end Dimention of the output (2D, 3D,...).
        @remarks *Might take alot of time, depends on the amount of data and its size.
        """       
        self.data = np.array(data, dtype=np.uint8)
        self.dim = dim
        logging.debug("initialzing ComputePCA")

    def run(self):
        """
        @return C{np.ndarray} -> the data in the output dim
        @remarks *Will compute PCA with self.data & self.dim.
                 *migh run into a MemoryError or ResourceExhausted Cuda Error if not enough VRAM is Avilable.
        """
        logging.info("Computing PCA, might take a while")
        Xn, means = self.normalize(self.data)
        Cov = np.matmul(np.transpose(Xn),Xn)
        Xtf = tf.placeholder(tf.float32, shape=[self.data.shape[0], self.data.shape[1]])
        Covtf = tf.placeholder(tf.float32, shape=[Cov.shape[0], Cov.shape[1]])
        stf, utf, vtf = tf.svd(Covtf)
        tvtf = tf.slice(vtf, [0, 0], [self.data.shape[1], self.dim])

        Ttf = tf.matmul(Xtf, tvtf)
        Rtf = tf.matmul(Ttf, tvtf, transpose_b=True)

        with tf.Session() as sess:
            Rn = sess.run(Rtf, feed_dict = {
                Xtf: Xn,
                Covtf: Cov
            })

        R = np.array(self.denormalize(Rn, means))
        logging.info("Finished Computing PCA")
        return R
    
    def normalize(self, X: np.ndarray):
        """
        @param X: C{np.ndarray} -> the input data
        @return C{tuple} -> The Normalized data
        """
        means = np.mean(X, axis=0)
        tmp = np.subtract(X, means)
        return tmp, means

    def denormalize(self, Rn : np.ndarray, means : np.ndarray):
        """
        @param Rn: C{np.ndarray} -> the Normalized data
        @param means: C{np.ndarray} -> the mean of the data
        @return C{tuple} -> The Denormalized data
        """
        return np.add(Rn, means)

class Visualize3D():
    """Visualize3D -> Scatter's Data on a 3D plot."""
    def __init__(self, data : np.ndarray, labels : list=None, save : bool=True, clustering : str="kmeans"):
        """
        @param data: C{np.ndarray} -> A np.array of [x, y ,z], uasally ComputePCA output.
        @param labels: C{list} -> [Optional], labels(aka Clusters) from Some kind of @TensorflowBaseWraper
        @param save: C{bool} -> [Optional], Should we save the data & Labels to npy files, Could be loaded and visualized Again.
        @param clustering: C{str} -> [Optional], Currnetly only used for filenameing, might be used for logging or smth.
        @remarks *self._save() is being called on inilization.
        """
        self.data = data
        self.labels = labels
        self.save = save
        self.clustering = clustering
        logging.debug("initialzing Visualize3D")
        self._save()

    def _save(self):
        """
        @remarks: *Save self.data & self.labels to .npy Files for future use.
                  *Can be loaded via @IntergrationSuit.visualize_from_file() OR @self.load() -> Should not be USED!
        """
        if self.labels is not None and self.data is not None and self.save:        
            filenameX = f"X_PCA_{len(self.data)}_{self.clustering}_3D.npy"
            filenameY = f"Y_PCA_{len(self.labels)}_{self.clustering}_3D.npy"
            logging.info(f"Saving PCA X Data to: {filenameX}")
            np.save(filenameX, self.data)
            logging.info(f"Saving PCA Y Data to: {filenameY}")
            np.save(filenameY, self.labels)

    def load(self, filename : str):
        """
        @param filename: The base Filename (aka PCA_80_kmeans_3D.npy)
        @remarks *it will look for X_filename & Y_filename, (DON'T Specify 'X_' or 'Y_')!
        """
        logging.info(f"Loading X_{filename} & Y_{filename}")
        self.data = np.load(f"X_{filename}")
        self.labels = np.load(f"Y_{filename}")
        logging.info("You can call Visualize3D.show()")

    def show(self):
        """
        @remarks *Will Scatter self.data on a 3D plot.
                 *if labels were supplied the will be attched to the legend.
                 *There Are About ~900 Dont have More then ~900 Clusters..
                 *Colors defiition is in @common.COLORS.
                 *Might take some time if there are alot of data points.
        """
        logging.info("Plotting Data, might take a while..")
        fig = plt.figure(figsize=(17,17))
        ax = fig.add_subplot(111, projection='3d')
        if len(self.labels) <= len(COLORS):
            raise RuntimeWarning(f"there are not enough colors for all the labels, {len(self.labels)}.\nAdd Colors from https://coolors.co/")

        if self.labels is not None:
            for index, arr in enumerate(self.data):
                ax.scatter(arr[0], arr[1], arr[2], c=COLORS[self.labels[index]], label=f"cluster_{self.labels[index]}")
        else:
            for index, arr in enumerate(self.data):
                ax.scatter(arr[0], arr[1], arr[2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
       
        plt_handles, plt_labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(plt_labels, plt_handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
        logging.info("Visualize3D Finished")

def to_visualization_format(data : np.ndarray):
    """
    @param data: C{np.ndarray} -> an array on Image Objects.
    @remarks *Will Split an Image object into Image.data & image.cluster_n (aka X, y).
             *Make sure you call @IOWraper.marge_data() befor Visualizing.
             *RuntimeWarning will be raised if there are missing values.
    """
    logging.debug("Transforming data to visualization format")
    x = [np.resize(image.data, VISUALIZATION_VECTOR_SIZE ** 2) for image in data]
    y = [image.cluster_n for image in data if image.cluster_n is not None]
    if len(y) == 0:
        raise RuntimeWarning("Make sure you @IOWraper.marge_data(), no Labels are Avilable")
        exit(self, 1)
    return (x, y)