#!/usr/bin/env python3
# Author: Yuval Kaneti

#### IMPROTS ####
import os
import cv2
import numpy as np
from tqdm import tqdm

#### FUNCTIONS ####
def load_data(folder_path : str, resize=False, size=None) -> list:
    """
    """
    print(f"Loading {len(list(os.listdir(folder_path)))} Images")
    images = []    
    for image_name in tqdm(os.listdir(folder_path)):    
        # images.append(io.imread(os.path.join(folder_path, image_name)))
        if resize and size is not None:
            image_path = os.path.join(folder_path, image_name)
            cv2_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            images.append(np.resize(cv2_image, (size, size)).flatten())
        else:
            images.append(np.asarray(cv2.cvtColor(cv2.imread(os.path.join(folder_path, image_name)), cv2.COLOR_BGR2RGB).flatten()))

    return np.asarray(images)   

def load_test_labels(folder_path : str) -> list:
    """

    """
    labels = []
    for filename in os.listdir(folder_path):
        if "1_" in filename:
            labels.append("r")
        else:
            labels.append("b")
    return labels


def save_data(folder_path : str, data : list) -> None:
    """
    """
    print(f"saving image to {folder_path}")
    cv2.imwrite(folder_path, data)
