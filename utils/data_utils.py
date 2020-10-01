#!/usr/bin/env python3
# Author: Yuval Kaneti

#### IMPROTS ####
import os
import cv2
import numpy as np
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.INFO)
from utils.common import IMAGE_SIZE
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#### FUNCTIONS ####
def data_to_np_image(data):
    return [tup[1] for tup in data]

# def load_data(folder_path , resize=False, size=IMAGE_SIZE) :
#     """
#     """
#     print(f"Loading {len(list(os.listdir(folder_path)))} Images")
#     images = []    
#     for image_name in tqdm(os.listdir(folder_path)):    
#         if resize and size is not None:
#             image_path = os.path.join(folder_path, image_name)
#             np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
#             np_image = np.resize(np_image, (size, size, 3))
#             # np_image = np.expand_dims(np_image, axis=0)
#             images.append((image_path, np_image))
#         else:
#             image_path = os.path.join(folder_path, image_name)
#             np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
#             # np_image = np.expand_dims(np_image, axis=0)
#             images.append((image_path, np_image))
# # 
#     return images    

def load_data(folder_path , resize=False, size=IMAGE_SIZE):
    logging.info(f"Loading images from directory : {folder_path}")
    images = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = load_img(image_path, target_size=(size, size))
        image_tensor = img_to_array(image)
        image_tensor = np.expand_dims(image_tensor, axis=0)
        image_tensor = preprocess_input(image_tensor)
        images.append((image_name, image_tensor))

    return [images]

def save_data(folder_path, data):
    """
    """
    logging.info(f"saving image to {folder_path}")
    cv2.imwrite(folder_path, data)
