import sys; sys.path.insert(0,'..')
import os
import matplotlib.pyplot as plt

import numpy as np

from common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from data_utils import load_data, save_data


def main():
    data = load_data(os.path.join(DATA_FOLDER_PATH, TEST_IMAGES_FOLDER))
    print(data.shape)



if __name__ == '__main__':
    main()