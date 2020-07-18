#!/usr/bin/env python3
# Author: Yuval Kaneti

#### IMPORTS ####
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn.manifold import TSNE

from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from utils.data_utils import load_data, load_test_labels

#### FUNCTIONS ####
def kdim_to_ndim(array, dim=2):
    return TSNE(n_components=dim).fit_transform(array)

def plot_3d(array, labels = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if labels is not None:
        for pos, lable in zip(array, labels):
            xs, ys, zs = pos
            ax.scatter(xs, ys, zs, marker="o", c = lable)

    else:
        for  xs, ys, zs in array:
            ax.scatter(xs, ys, zs, marker="o", c = 'b')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_2d(array, labels = None):
    if labels is not None:
        for pos, lable in zip(array, labels):
            print(pos, lable)
            xs, ys = pos
            plt.scatter(xs, ys, marker="o", c = lable)
    else:
        for xs, ys in array:
            plt.scatter(xs, ys, marker="o", c = "b")

    plt.show()

def main():
    TEST = False
    DIM = 3
    if TEST:
        PATH = os.path.join(DATA_FOLDER_PATH, TEST_IMAGES_FOLDER)
        data = load_data(PATH)
        print(data.shape)
        transformed = kdim_to_ndim(data, dim = DIM)
        print(transformed)
        if DIM == 2:
            plot_2d(transformed, labels = load_test_labels(PATH))
        else:
            plot_3d(transformed, labels = load_test_labels(PATH))
    else:
        PATH = os.path.join(DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER)
        data = load_data(PATH)
        print(data.shape)
        transformed = kdim_to_ndim(data, dim = DIM)
        print(transformed)
        if DIM == 2:
            plot_2d(transformed)
        else:
            plot_3d(transformed)  

if __name__ == '__main__':
    main()