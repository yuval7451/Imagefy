#! /usr/bin/env python3
"""
Author: Yuval Kaneti⭐
Code Taken From: https://github.com/titu1994/neural-image-assessment/blob/master/
## Example Usage ##
image_dir = "YOUR\\IMAGE\\DIR"
Inception = InceptionResnetWraper(gpu=True, _verbose=True) -> Only If CuDNN Installed Use GPU Else gpu=False
images = Inception.load_images(image_dir)
scores = Inception.predict(images)
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

class InceptionResnetWraper(object):
    """
    InceptionResnetWraper: Is a Wraper Above a Pre-Trained InceptionResNetV2 on @AVA2 Dataset 
    """
    #### FUNCTIONS ####
    def __init__(self, gpu : bool, verbose : bool):
        """
        :type
        """  
        self.weights_path = os.path.join(WEIGHTS_FOLDER_PATH, INCEPTION_RESNET_WEIGHTS)
        self.target_size = (IMAGE_SIZE, IMAGE_SIZE)
        self._verbose = verbose
        self.model = self._load_modal(gpu=gpu)


    def _load_modal(self, gpu : bool):
        """
        :type gpu: C{bool} -> Should Keras Use a GPU or CPU
        :type _verbose: {bool} -> Vebosity
        :return model: C{keras.models.Model} -> The Loaded Model
        """
        if gpu:
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


    def load_images(self, image_dir : str):
        """
        :type image_dir: C{str} -> The Image Directory
        :return images: C{list} -> A List Containing Tuples (image_name, image_tensor)
        """
        if self._verbose:
            print("Loading images from directory : ", image_dir)
        images = []
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            image = load_img(image_path, target_size=self.target_size)
            image_tensor = img_to_array(image)
            image_tensor = np.expand_dims(image_tensor, axis=0)
            image_tensor = preprocess_input(image_tensor)
            images.append((image_name, image_tensor))
    
        return images
 
 
    def predict(self, images : list):
        """
        :type images: C{list} -> The list returned from @self.load_images See @self.load_images.__doc__ For More Info
        :return score_list: C{list} -> A list of Tuples (image_name, mean), *mean of the socres
        """
        with tf.device(self.device):
            score_list = []
            for image_name, tensor in images:
                scores = self.model.predict(tensor, batch_size=1, verbose=0)[0]
                mean = mean_score(scores)
                std = std_score(scores)
                score_list.append((image_name, mean))
                if self._verbose:
                    print("Evaluating : ", image_name)
                    print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))

        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        return score_list

    def rank(self, score_list : list):
        """
        :type score_list: C{list} -> The list returned from @self.predict See @self.predict.__doc__ For More Info
        """
        print("*" * 40, "Ranking Images", "*" * 40)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(score_list):
            print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))        



image_dir = "test"
Inception = InceptionResnetWraper(gpu=True, verbose=True)
# images = Inception.load_images(image_dir)
images = load_data(image_dir, resize=True, size=IMAGE_SIZE)
images = pipeline_to_tensor(images)
scores = Inception.predict(images)
Inception.rank(scores)
print(scores)