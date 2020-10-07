#! /usr/bin/env python3
"""
Author: Yuval Kaneti
"""

#### Imports ####
import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import numpy as np
import tensorflow as tf
from imagefy_migration.wrapers.base_wraper import BaseWraper
from imagefy_migration.utils.data_utils import  WraperOutput
from imagefy_migration.utils.score_utils import mean_score
from imagefy_migration.utils.common import IFERENCE_MODEL_DIR, INCEPTION_RESNET_INFERENCE_INPUT, INCEPTION_RESNET_INFERENCE_DENSE
from imagefy_migration.wrapers.config import InceptionConfig
from pprint import pformat

class InceptionResnetTensorflowWraper(BaseWraper):
    """InceptionResnetTensorflowWraper -> An implemntion of InceptionResnetV2 in Tensorflow."""
    def __init__(self, **kwargs: dict):
        """
        @param num_epochs: C{int} -> The number of Training epochs.
        @param num_clusters: C{int} -> The number of Clusters.
        @param batch_size: C{int} -> The Batch size.
        @param kwargs: C{dict} -> For futre use.
        @local config: C{tf.compat.v1.ConfigProto} -> the config & hooks handler for the estimator.
        """
        super().__init__(**kwargs) 
        self.inference_model_dir = IFERENCE_MODEL_DIR # FIXME?
        self.config = InceptionConfig()
        self.predictor = tf.saved_model.load(self.inference_model_dir) 
        
    def run(self):
        """
        @remarks *This is where the action starts.
        @return C{InceptionResnetTensorflowWraperOutput} -> the clustering result, used for IOWraper.
        """     
        logging.info(f"Starting {self.name}")
        self.wraper_output = self.predict()

        if self._use_tensorboard:
            logging.warn(f"Tensorboard is not Implemented in {self.name}")
        
        return self.wraper_output

    def predict(self):
        """
        @return C{InceptionResnetTensorflowWraperOutput} -> the Predicted output, will be used for IOWraper.
        @remarks *Loads the inference model and predicts images score.
        """
        self.dataset = self._input_fn()
        logging.info("Starting to make prediction")
        predictor_output_list = []
        for (image, label, image_name) in self.dataset:
            (image, label, image_name) = self.pre_process_data(image.numpy(), label.numpy(), image_name.numpy())
            output_dict = self._predict_one(image)
            y_predicted = output_dict[INCEPTION_RESNET_INFERENCE_DENSE][0].numpy()
            score = mean_score(y_predicted)
            predictor_output = PredictorOutput(label=label, image_name=image_name, score=score, index=len(predictor_output_list))
            logging.debug(predictor_output)
            predictor_output_list.append(predictor_output)
                
        logging.info("Finished making predictions")

        return InceptionResnetTensorflowWraperOutput(predictor_output_list=predictor_output_list)

    def _predict_one(self, image):
        """
        @param image: C{list} -> A Flattened Image.
        @return C{np.ndarray} -> A numpy array length 10 containing the score.
        """
        with tf.device('/device:GPU:0'): 
            example = tf.train.Example()
            example.features.feature[INCEPTION_RESNET_INFERENCE_INPUT].float_list.value.extend(image)
            return self.predictor.signatures["serving_default"](predictor_inputs=tf.constant([example.SerializeToString()]))

    def pre_process_data(self, image: np.ndarray, label: bytes, image_name: bytes):
        """
        @param image: C{np.ndarray} -> The Image data .
        @param label: C{bytes} -> The image source cluster.
        @param image_name: C{bytes} -> The image name.
        @return C{tuple} -> the pre proccesed data
        """
        label = label.decode()  
        image_name = image_name.decode() 
        image = image.flatten().tolist()
        return (image, label, image_name)

class PredictorOutput():
    """PredictorOutput -> A Deserialize output from a tf.contrib.predictor.from_saved_model(...)."""
    def __init__(self, label: str, image_name: str, score: float, index: int):
        """
        @param label: C{str} -> The cluster name that the image generated this output is in.
        @param image_name: C{str} -> the image name this output represents.
        @param score: C{float} -> The score the image got.
        @param index: C{index} -> It's index in the original order, used for IOWraper.
        """
        self.label = label
        self._label = int(self.label.split("_")[-1])
        self.image_name = image_name
        self.score = float(score)
        self.top = False
        self._index = int(index)

    def __str__(self):
        return f"label: {self.label}, image_name: {self.image_name}, score: {self.score}, index: {self._index}"
    
    def __repr__(self):
        return f"label: {self.label}, image_name: {self.image_name}, score: {self.score}, index: {self._index}"
        
class InceptionResnetTensorflowWraperOutput(WraperOutput):
    """InceptionResnetTensorflowWraperOutput -> A WraperOutput Object for InceptionResnetTensorflowWraper."""
    def __init__(self, predictor_output_list: list):
        """
        """
        self.predictor_output_list = predictor_output_list
        self.cluster_labels = [predictor_output._label for predictor_output in self.predictor_output_list]
        self.grouped_outputs = self._group()
        self.sorted_outputs = self._sort()
        n_clusters = max(self.cluster_labels) + 1
        super().__init__(n_clusters, self.cluster_labels)

    def __str__(self):
        return_string = ""
        for output in self.predictor_output_list:
            return_string += str(output) + "\n"
        return return_string

    def _group(self):
        """
        @return C{dict} -> A dict where each key is a label and a value is a list of PredictorOutput Objects.
        @remarks *Groups PredictorOutput Objects by their label.
        """
        grouped_outputs = {}
        for predictor_output in self.predictor_output_list:
            if predictor_output.label in grouped_outputs.keys():
                grouped_outputs[predictor_output.label].append(predictor_output)
            else:
                grouped_outputs[predictor_output.label] = [(predictor_output)]
        return grouped_outputs

    def _sort(self):
        """
        @return C{dict} -> A dict where each key is a label and a value is a list of PredictorOutput Objects Sorted by their Score.
        @remarks *Sorts PredictorOutput Objects by their Score.
        """
        sorted_outputs = {}
        for label, group in self.grouped_outputs.items():
            sorted_outputs[label] = sorted(group, key=lambda x: x.score, reverse=True)

        return sorted_outputs

    def _order(self, outputs: list):
        """
        @param output: C{list} -> The output of self.top(...), A list of PredictorOutput, some will have .top=True.
        @return C{list} -> A list of PredictorOutput Objects ordered by the .index variable.
        @remarks *Reorders a list of PredictorOutput by their .index variable. 
        """
        ordered_outputs = [None for i in range(len(outputs))]
        for output in outputs:
            ordered_outputs[output._index] = output
        
        return ordered_outputs

    def top(self, k: int):
        """
        @param k: C{int} -> the number of element to consider as top from each cluster.
        @returns C{list} -> A list of PredictorOutput, while the top ones haveing their top=True, used in IOWraper.
        @remarks *Given a k parameter it takes the top K PredictorOutput from every cluster by their score and sets their top=True. 
        """
        outputs = []
        for label, group in self.sorted_outputs.items():
            for index, output in enumerate(group):
                if index < k:
                    output.top = True
                outputs.append(output)
        
        return self._order(outputs)

