#! /usr/bin/env python3
"""
Code Taken From: https://github.com/tensorflow/tensorflow/issues/20942 @kevintrankt commented on Jul 23, 2018
Code Taken from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
Wiki: Silhouette clustering -> https://en.wikipedia.org/wiki/Silhouette_(clustering)
Author: Yuval Kaneti⭐
"""

#### Imports ####
import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

import tensorflow as tf
from imagefy.wrapers.base_wraper import BaseWraper
from imagefy.utils.data_utils import  WraperOutput
from imagefy.suits.config import Config

class MiniBatchKmeansTensorflowWraper(BaseWraper):
    """MiniBatchKmeansTensorflowWraper -> An implemntion of Minibatch Kmeans in Tensorflow."""
    def __init__(self, num_epochs: int, num_clusters: int, batch_size: int, save: bool, **kwargs: dict):
        """
        @param data: C{list} -> a list of Image Objects.
        @param num_epochs: C{int} -> The number of Training epochs.
        @param num_clusters: C{int} -> The number of Clusters.
        @param save: C{bool} -> Should we Save the Trained Model, NO IMPLEMNTED YET.
        """
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self._save = save if not save else False; logging.warn(f"{self.name}.Save() is not implemnted yet.`")
        self.wraper_output = None
        self.config = Config(self.model_dir, self.base_model_dir)
        
    def run(self):
        """
        @remarks *This is where the action starts.
        """     
    
        self.cluster = tf.compat.v1.estimator.experimental.KMeans(
            num_clusters=self.num_clusters,
            use_mini_batch=False,
            config=self.config.get_run_config(),
            # mini_batch_steps_per_iteration = 10,
        )
        self._train(hooks=self.config.get_hooks()) 
        self.wraper_output = self._transform()
        if self._use_tensorboard:
            self._tensorboard()
        
        return self.wraper_output

    def _train(self, hooks: list):
        # train
        with tf.device('/gpu:0'):
            logging.info("Starting to train")
            # for _ in tqdm(range(self.num_epochs)):
            self.cluster.train(input_fn=lambda: self._input_fn(                                
                                    batch_size=self.batch_size,
                                    shuffle=False, 
                                    num_epochs=self.num_epochs),
                                    hooks=hooks)
                                    
            score = self.cluster.score(input_fn=lambda: self._input_fn(                                       
                                        batch_size=self.batch_size,
                                        shuffle=False, 
                                        num_epochs=self.num_epochs * 2))

            logging.info(f"score: {score}")
            logging.info("Saving Trainable Variables")
        
        # for value in  tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES):
        #     tf.summary.histogram(value.name, value)

    def _transform(self):
        with tf.device('/gpu:0'):
        # map the input points to their clusters
            logging.info("starting to Transform the data")
            cluster_indices = list(self.cluster.predict_cluster_index(input_fn=lambda: self._input_fn(                                                                  
                                                                    batch_size=self.batch_size,
                                                                    shuffle=False, 
                                                                    num_epochs=self.num_epochs)))
            logging.debug(f"There are {len(cluster_indices)} labels")                                        
        return MiniBatchKmeansWraperOutput(cluster_labels=cluster_indices)

    def save(self):
        # Saving estimator model
        model_path = os.path.join(self.model_dir, self.model_name)
        logging.info(f"Saving model to {model_path}")
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(self._input_fn())
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        tf.feature_column.make_parse_example_spec(feature_columns))
        self.cluster.estimator.export_saved_model(model_path, serving_input_fn)
    
    def load(self, path):
        logging.info(f"loading model from {path}")
        self.cluster = tf.saved_model.load(path)
    
    def _tensorboard(self):
        if self._loader.dtype is tf.data.Dataset:        
            logging.warn("Tensorboard is in BETA with Tensorload")
        super()._tensorboard(dtype=self._loader.dtype, batch_size=self.batch_size)

class MiniBatchKmeansWraperOutput(WraperOutput):
    """MiniBatchKmeansWraperOutput -> A WraperOutput Object for MiniBatchKemansTensorflowWraper."""
    def __init__(self, cluster_labels):
        n_clusters = max(cluster_labels) + 1
        super().__init__(n_clusters, cluster_labels)
