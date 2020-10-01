#! /usr/bin/env python3
"""
Author: Yuval Kaneti⭐
Code Taken From: https://github.com/titu1994/neural-image-assessment/blob/master/
## Example Usage ##
image_dir = "YOUR\\IMAGE\\DIR"
Inception = InceptionResnetWraper(gpu=True, _verbose=True) -> Only If CuDNN Installed Use GPU Else gpu=False
data = Inception.load_images(image_dir)
scores = Inception.predict(data)
"""

#### IMPROTS ####
import os
import numpy as np
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from utils.score_utils import mean_score, std_score
from utils.common import WEIGHTS_FOLDER_PATH, INCEPTION_RESNET_WEIGHTS, IMAGE_SIZE
from utils.data_utils import load_data
from development.base_wraper import BaseWraper

# from utils.common import DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER, TEST_IMAGES_FOLDER

class InceptionResnetWraper(BaseWraper):
    """
    InceptionResnetWraper: Is a Wraper Above a Pre-Trained InceptionResNetV2 on @AVA2 Dataset 
    """
    #### FUNCTIONS ####
    def __init__(self, data, gpu, verbose ):
        """
        """  
        self.weights_path = os.path.join(WEIGHTS_FOLDER_PATH, INCEPTION_RESNET_WEIGHTS)
        self._data = data
        self._gpu = gpu
        self._verbose = verbose
        self.target_size = (IMAGE_SIZE, IMAGE_SIZE)

    def start(self):
        """
        """
        self.model = self._load_modal()
        self._score_lists = self._predict()
        return self._score_lists

    def _load_modal(self):
        """
        :type gpu: C{bool} -> Should Keras Use a GPU or CPU
        :type _verbose: {bool} -> Vebosity
        :return model: C{tensorflow.keras.models.Model} -> The Loaded Model
        """
        if self._gpu:
            self.device = '/GPU:0'            
        else:
            self.device = '/CPU:0'
            
        if self._verbose:
            logging.debug(f"Using {self.device}")
            logging.info("Loading Model")
        with tf.device(self.device):
            base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
            # x = Dropout(0.75)(base_model.output)
            # x = Dense(10, activation='softmax')(x)
            x = Dense(10, activation='softmax')(base_model.output)
            model = Model(base_model.input, x)
            model.load_weights(self.weights_path)
            # model.save("../models/InceptionResnet.h5")
        return model
 
    def _predict(self):
        """
        :type data: C{list} -> The list returned from @self.load_images See @self.load_images.__doc__ For More Info
        :return score_list: C{list} -> A list of Tuples (image_path, mean), *mean of the socres
        """
        score_lists = []
        with tf.device(self.device):
            for folder in self._data:
                score_list = []
                for image_path, tensor in folder:
                    scores = self.model.predict(tensor, batch_size=1, verbose=0)[0]
                    mean = mean_score(scores)
                    std = std_score(scores)
                    score_list.append((image_path, mean))
                    if self._verbose:
                        logging.debug(f"Evaluating : {image_path}")
                        logging.debugf"NIMA Score : {mean}, {std}")
                
                
                score_list = sorted(score_list, key=lambda x: x[1], reverse=True)  
                   
                self._rank(score_list)
                       
                score_lists.append(score_list)
        logging.debug(len(score_lists))                                   
        return score_lists

    def _rank(self, score_list):
        """
        """
        rank_msg = "*" * 40, "Ranking Images", "*" * 40
        logging.info(rank_msg)    
        for i, (name, score) in enumerate(score_list):
            logging.info(f"{i+1}), {name} = {score}")
            # %d)" % (i + 1), "%s : Score = %0.5f" % (name, score))        


# def main():
#     TEST = True
#     if TEST:
#         PATH = os.path.join(DATA_FOLDER_PATH, TEST_IMAGES_FOLDER)
#     else:
#         PATH = os.path.join(DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER)

#     data = load_data(PATH,resize=True)

#     inception = InceptionResnetWraper(data, True, True)
#     print(inception.start())
    
# if __name__ == '__main__':
#     main()