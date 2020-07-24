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
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from utils.score_utils import mean_score, std_score
from utils.common import WEIGHTS_FOLDER_PATH, INCEPTION_RESNET_WEIGHTS, IMAGE_SIZE
from utils.data_utils import load_data, pipeline_to_tensor
from Development.base_wraper import BaseWraper

class InceptionResnetWraper(BaseWraper):
    """
    InceptionResnetWraper: Is a Wraper Above a Pre-Trained InceptionResNetV2 on @AVA2 Dataset 
    """
    #### FUNCTIONS ####
    def __init__(self, data : list, gpu : bool, verbose : bool):
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
        self._score_list = self._predict()

    def _load_modal(self):
        """
        :type gpu: C{bool} -> Should Keras Use a GPU or CPU
        :type _verbose: {bool} -> Vebosity
        :return model: C{keras.models.Model} -> The Loaded Model
        """
        if self._gpu:
            self.device = '/GPU:0'            
        else:
            self.device = '/CPU:0'
            
        if self._verbose:
            print(f"Using {self.device}")
        with tf.device(self.device):
            base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
            x = Dropout(0.75)(base_model.output)
            x = Dense(10, activation='softmax')(x)
            model = Model(base_model.input, x)
            model.load_weights(self.weights_path)
            
        return model
 
    def _predict(self):
        """
        :type data: C{list} -> The list returned from @self.load_images See @self.load_images.__doc__ For More Info
        :return score_list: C{list} -> A list of Tuples (image_path, mean), *mean of the socres
        """
        with tf.device(self.device):
            score_list = []
            for image_path, tensor in self._data:
                scores = self.model.predict(tensor, batch_size=1, verbose=0)[0]
                mean = mean_score(scores)
                std = std_score(scores)
                score_list.append((image_path, mean))
                if self._verbose:
                    print("Evaluating : ", image_path)
                    print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
               
               
            score_list = sorted(score_list, key=lambda x: x[1], reverse=True)     
            if self._verbose:
                self._rank(score_list)   
                                         
        return score_list

    def _rank(self, score_list):
        """
        """
        print("*" * 40, "Ranking Images", "*" * 40)
        for i, (name, score) in enumerate(score_list):
            print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))        

