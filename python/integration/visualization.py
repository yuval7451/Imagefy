"""
"""

import time
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from integration.common import VISUALIZATION_VECTOR_SIZE

class ComputePCA():
    """
    """
    def __init__(self, data, dim):
        """
        """       
        self.data = np.array(data, dtype=np.uint8)
        self.dim = dim
        logging.debug("initialzing ComputePCA")


    def run(self):
        """
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
    
    def normalize(self, X):
        """
        """
        means = np.mean(X, axis=0)
        tmp = np.subtract(X, means)
        return tmp, means

    def denormalize(self, Rn, means):
        """
        """
        return np.add(Rn, means)

class Visualize3D():
    """
    """
    def __init__(self, data, labels=None, save=True, clustering="kmeans"):
        """
        """
        self.data = data
        self.labels = labels
        self.save = save
        self.clustering = clustering
        logging.debug("initialzing Visualize3D")
        self._save()

    def _save(self):
        if self.labels is not None and self.data is not None and self.save:        
            filenameX = f"X_PCA_{len(self.data)}_{self.clustering}_3D.npy"
            filenameY = f"Y_PCA_{len(self.labels)}_{self.clustering}_3D.npy"
            logging.info(f"Saving PCA X Data to: {filenameX}")
            np.save(filenameX, self.data)
            logging.info(f"Saving PCA Y Data to: {filenameY}")
            np.save(filenameY, self.labels)

    def load(self, filename):
        logging.info(f"Loading X_{filename} & Y_{filename}")
        self.data = np.load(f"X_{filename}")
        self.labels = np.load(f"Y_{filename}")
        logging.info("You can call Visualize3D.show()")

    def show(self):
        """
        """
        logging.info("Plotting Data, might take a while..")
        # color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        color = ['#EF6461', '#313638', '#E0DFD5', '#E4B363', '#A8B4A5', '#725D68', '#9D6A89', '#EE92C2', 
                 '#EFBC9B', '#492C1D', '#5B5750', '#6B7F82', '#8EB8E5', '#F9B4ED', '#E574BC','#C52184',
                  '#1E2D24', '#334139', '#F22B29', '#140F2D', '#F49D37', '#3F88C5', '#D72638', '#81D2C7',
                  '#AFECE7', '#8D99AE', '#8D99AE', '#2B2D42', '#F22B29']

        fig = plt.figure(figsize=(17,17))
        ax = fig.add_subplot(111, projection='3d')
        # print(self.labels)
        if len(self.labels) <= len(color):
            raise RuntimeWarning(f"there are not enough colors for all the labels, {len(self.labels)}.\nAdd Colors from https://coolors.co/")

        if self.labels is not None and len(self.labels) <= len(color):
            for index, arr in enumerate(self.data):
                ax.scatter(arr[0], arr[1], arr[2], c=color[self.labels[index]], label=f"cluster_{self.labels[index]}")
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


def to_visualization_format(data):
    """
    """
    logging.debug("Transforming data to visualization format")
    x = [np.resize(image.data, VISUALIZATION_VECTOR_SIZE ** 2) for image in data]
    y = [image.cluster_n for image in data if image.cluster_n is not None]
    if len(y) == 0:
        raise RuntimeWarning("Make sure you @IOWraper.marge_data(), no Labels are Avilable")
        exit(self, 1)
    return (x, y)