#! /usr/bin/env python3
"""
Author: Yuval Kaneti⭐
"""

#### IMPROTS ####
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from utils.common import WEIGHTS_FOLDER_PATH, INCEPTION_RESNET_WEIGHTS

# from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER

def export_model(weights_path):
    with tf.device('/CPU:0'):
        base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
        # x = Dense(256, activation='relu')(base_model.output)
        x = Dense(10, activation='softmax')(base_model.output)
        model = Model(base_model.input, x)
        model.load_weights(weights_path)
        model.save("C:\\yuval\\computer\\Imagefy\\models\\InceptionResnet")
    return model


def main():
    weights_path = os.path.join(WEIGHTS_FOLDER_PATH, INCEPTION_RESNET_WEIGHTS)
    export_model(weights_path)
    
if __name__ == '__main__':
    main()