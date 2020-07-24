#!/usr/bin/env python3
# Author: Yuval Kaneti

#### IMPROTS ####
import os
import cv2
import numpy as np
from tqdm import tqdm

from utils.common import IMAGE_SIZE
#### FUNCTIONS ####
def data_to_np_image(data):
    return [tup[1] for tup in data]


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
            np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            np_image = np.resize(np_image, (size, size, 3))
            images.append((image_path, np_image))
        else:
            image_path = os.path.join(folder_path, image_name)
            np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            images.append((image_path, np_image))
# 
    return np.asarray(images)     
    

def save_data(folder_path, data):
    """
    """
    print(f"saving image to {folder_path}")
    cv2.imwrite(folder_path, data)
