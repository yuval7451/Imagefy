#!/usr/bin/env python3
# Author: Yuval Kaneti

#### IMPROTS ####
import os
import cv2
import numpy as np
from tqdm import tqdm

from utils.common import IMAGE_SIZE
#### FUNCTIONS ####
def pipeline_to_cluster(data):
    return [tup[1] for tup in tqdm(data)]


def pipeline_to_tensor(data):
    return [(tup[0], np.expand_dims(tup[1].reshape(IMAGE_SIZE, IMAGE_SIZE, 3), axis=0)) for tup in tqdm(data)]
    
def load_data(folder_path : str, resize=False, size=IMAGE_SIZE) :
    """
    """
    print(f"Loading {len(list(os.listdir(folder_path)))} Images")
    images = []    
    for image_name in tqdm(os.listdir(folder_path)):    
        if resize and size is not None:
            image_path = os.path.join(folder_path, image_name)
            cv2_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            images.append((image_path, np.resize(cv2_image, (size, size, 3)).flatten()))
        else:
            image_path = os.path.join(folder_path, image_name)
            cv2_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            images.append((image_path, cv2_image.flatten()))
# 
    return np.asarray(images)     
    
    

def load_test_labels(folder_path : str):
    """

    """
    labels = []
    for filename in os.listdir(folder_path):
        if "1_" in filename:
            labels.append("r")
        else:
            labels.append("b")
    return labels


def save_data(folder_path : str, data : list):
    """
    """
    print(f"saving image to {folder_path}")
    cv2.imwrite(folder_path, data)
