#! /usr/bin/env python3
"""
Author: Yuval Kaneti⭐
"""

#### Imports ####
import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import numpy as np
import tensorflow as tf
from integration.suits.config import Config
from integration.wrapers.base_wraper import BaseWraper
from integration.utils.data_utils import  WraperOutput
from integration.utils.score_utils import mean_score, std_score
from integration.utils.common import INCEPTION_RESNET_INFERENCE_INPUT, PREDICTOR_INFERENCE_INPUTS, INCEPTION_RESNET_INFERENCE_DENSE


class InceptionResnetTensorflowWraper(BaseWraper):
    """InceptionResnetTensorflowWraper -> An implemntion of InceptionResnetV2 in Tensorflow."""
    def __init__(self, **kwargs: dict):
        """
        @param num_epochs: C{int} -> The number of Training epochs.
        @param num_clusters: C{int} -> The number of Clusters.
        @param batch_size: C{int} -> The Batch size.
        @param kwargs: C{dict} -> For futre use.
        @local config: C{config.Config} -> the config & hooks handler for the estimator.
        """
        super().__init__(**kwargs) 
        self.iterator = tf.compat.v1.data.make_one_shot_iterator(self._input_fn())
        self.inference_model_dir = "D:\\Imagefy\\resources\\models\\InceptionResNetV2\\inference\\1601923417"
        self.config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                    allow_growth=True,
                    force_gpu_compatible=True,
                    per_process_gpu_memory_fraction=0.9,
                    ),
            allow_soft_placement=True,
            log_device_placement=False,
        )
        
    def run(self):
        """
        @remarks *This is where the action starts.
        @return C{InceptionResnetTensorflowWraperOutput} -> the clustering result, used for IOWraper.
        """     
        self.wraper_output = self._predict()

        if self._use_tensorboard:
            raise NotImplementedError(f"Tensorboard is not Implemented in {self.name}")
            # self._tensorboard(batch_size=self.batch_size)
        
        return self.wraper_output

    def _predict(self):
        predictor_output_list = []
        next_element = self.iterator.get_next()
        with tf.compat.v1.Session(config=self.config) as sess:
            try:
                # tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.inference_model_dir)
                predictor = tf.contrib.predictor.from_saved_model(self.inference_model_dir)
                while True:
                    with tf.device('/device:GPU:0'):    
                        image, label, image_name = sess.run(next_element)
                        (image, label, image_name) = self.pre_process_data(image, label, image_name)
                        model_input = tf.train.Example(features=tf.train.Features(
                            feature={INCEPTION_RESNET_INFERENCE_INPUT: tf.train.Feature(
                                float_list=tf.train.FloatList(value=image))}))

                        model_input = model_input.SerializeToString()
                        output_dict = predictor({PREDICTOR_INFERENCE_INPUTS: [model_input]})
                        y_predicted = output_dict[INCEPTION_RESNET_INFERENCE_DENSE][0]
                    
                    score = mean_score(y_predicted)
                    # std = std_score(y_predicted)
                    predictor_output = PredictorOutput(label=label, image_name=image_name, score=score, index=len(predictor_output_list))
                    logging.info(predictor_output)
                    predictor_output_list.append(predictor_output)
                   
            except tf.errors.OutOfRangeError:
                logging.info("Finished Loading Data")

        return InceptionResnetTensorflowWraperOutput(predictor_output_list=predictor_output_list)


    def pre_process_data(self, image: np.ndarray, label: bytes, image_name: bytes):
        label = label.decode()  
        image_name = image_name.decode() 
        # logging.debug(f"label: {label}, Image name: {image_name}")
        image = image.flatten().tolist()
        return (image, label, image_name)

    
class PredictorOutput():
    """PredictorOutput -> A Deserialize output from a tf.contrib.predictor.from_saved_model(...)."""
    def __init__(self, label: str, image_name: str, score: float, index: int):
        self.label = int(label.split("_")[-1])
        self.image_name = image_name
        self.score = float(score)
        self.top = False
        self._index = int(index)

    def __str__(self):
        return f"label: {self.label}, image_name: {self.image_name}, score: {self.score}, index: {self._index}"

class InceptionResnetTensorflowWraperOutput(WraperOutput):
    """InceptionResnetTensorflowWraperOutput -> A WraperOutput Object for InceptionResnetTensorflowWraper."""
    def __init__(self, predictor_output_list: list):
            self.predictor_output_list = predictor_output_list
            self.cluster_labels = [predictor_output.label for predictor_output in self.predictor_output_list]
            self.grouped_outputs = self._group()
            self.sorted_outputs = self._sort()
            # self.outputs = self.top()
            n_clusters = max(self.cluster_labels) + 1
            super().__init__(n_clusters, self.cluster_labels)

    def __str__(self):
        return_string = ""
        for output in self.predictor_output_list:
            return_string += str(output) + "\n"
        return return_string

    def _group(self):
        # logging.debug("Group")
        grouped_outputs = {}
        for predictor_output in self.predictor_output_list:
            if predictor_output.label in grouped_outputs.keys():
                grouped_outputs[predictor_output.label].append(predictor_output)
            else:
                grouped_outputs[predictor_output.label] = [(predictor_output)]
        # logging.debug(grouped_outputs)
        return grouped_outputs

    def _sort(self):
        # logging.debug("Sort")
        sorted_outputs = {}
        for label, group in self.grouped_outputs.items():
            logging.debug(label)
            logging.debug(group)
            sorted_outputs[label] = sorted(group, key=lambda x: x.score, reverse=True)
            #group.sort(key=lambda x: x.score, reverse=True)

        # logging.debug(sorted_outputs)
        return sorted_outputs

    def _order(self, outputs: list):
        # logging.debug("Order")
        ordered_outputs = [None for i in range(len(outputs))]
        for output in outputs:
            ordered_outputs[output._index] = output

        # logging.debug(ordered_outputs)
        return ordered_outputs

    def top(self, k: int=3):
        logging.debug("Top")
        outputs = []
        for label, group in self.sorted_outputs.items():
            for index, output in enumerate(group):
                if index < k:
                    logging.debug(f"top: {output.image_name} label: {label}")
                    output.top = True
                outputs.append(output)

        # logging.debug(outputs)
        return self._order(outputs)

