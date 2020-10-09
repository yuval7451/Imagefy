#!/usr/bin/env python3
# Author: Yuval Kaneti


#### Imports ####
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
from imagefy.wrapers.config import KmeansConfig
from imagefy.wrapers.base_wraper import BaseWraper
from imagefy.utils.data_utils import  WraperOutput

class MiniBatchKmeansTensorflowWraper(BaseWraper):
    """MiniBatchKmeansTensorflowWraper -> An implemntion of Minibatch Kmeans in Tensorflow."""
    def __init__(self, num_epochs: int, num_clusters: int, batch_size: int, **kwargs: dict):
        """
        @param num_epochs: C{int} -> The number of Training epochs.
        @param num_clusters: C{int} -> The number of Clusters.
        @param batch_size: C{int} -> The Batch size.
        @param kwargs: C{dict} -> For futre use.
        @local config: C{config.Config} -> the config & hooks handler for the estimator.
        """
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.config = KmeansConfig(self.base_model_dir)

    def run(self):
        """
        @remarks *This is where the action starts.
        @return C{MiniBatchKmeansWraperOutput} -> the clustering result, used for IOWraper.
        """     
        self.cluster = tf.compat.v1.estimator.experimental.KMeans(
            num_clusters=self.num_clusters,
            use_mini_batch=False,
            config=self.config.get_run_config(),
        ) 

        self._train(hooks=[])
        self.wraper_output = self._transform()
        if self._use_tensorboard:
            self._tensorboard(batch_size=self.batch_size)
        
        return self.wraper_output

    @tf.function
    def _train(self, hooks: list):
        """
        @param hooks: C{list} -> A list of hooks for training.
        @remarks *Trains the estimator.
        """
        with tf.device('/device:GPU:0'): #GPU
            logging.info("Starting to train")
            self.cluster.train(input_fn=lambda: self._input_fn(                                
                                    batch_size=self.batch_size,
                                    shuffle=False, 
                                    num_epochs=self.num_epochs),
                                    hooks=hooks)
                                    
            self.score = self.cluster.score(input_fn=lambda: self._input_fn(                                       
                                        batch_size=self.batch_size,
                                        shuffle=False, 
                                        num_epochs=self.num_epochs))

            logging.info(f"score: {self.score}")
    
    def _transform(self):
        """
        @remarks *splits the input data to each cluster.
        @return C{MiniBatchKmeansWraperOutput}  -> The clustering output, used for IOWraper.
        """
        with tf.device('/device:gpu:0'):
            # map the input points to their clusters
            logging.info("starting to Transform the data")
            cluster_indices = list(self.cluster.predict_cluster_index(input_fn=lambda: self._input_fn(                                                                  
                                                                    batch_size=self.batch_size,
                                                                    shuffle=False, 
                                                                    num_epochs=self.num_epochs)))
        logging.debug(f"There are {len(cluster_indices)} labels")       
        return MiniBatchKmeansWraperOutput(cluster_labels=cluster_indices)
    
class MiniBatchKmeansWraperOutput(WraperOutput):
    """MiniBatchKmeansWraperOutput -> A WraperOutput Object for MiniBatchKemansTensorflowWraper."""
    def __init__(self, cluster_labels):
        n_clusters = max(cluster_labels) + 1
        super().__init__(n_clusters, cluster_labels)
