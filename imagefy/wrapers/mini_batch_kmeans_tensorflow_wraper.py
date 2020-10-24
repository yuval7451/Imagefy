# Author: Yuval Kaneti

## Imports
import os
from typing import List; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging 
import tensorflow as tf
from imagefy.wrapers.config import KmeansConfig
from imagefy.wrapers.base_wraper import BaseWraper
from imagefy.utils.data_utils import  WraperOutput

class MiniBatchKmeansTensorflowWraper(BaseWraper):
    """MiniBatchKmeansTensorflowWraper -> An implemntion of Minibatch Kmeans in Tensorflow."""
    def __init__(self, num_epochs: int, num_clusters: int, batch_size: int, filenames: List[str], **kwargs: dict):
        """
        @param num_epochs: C{int} -> The number of Training epochs.
        @param num_clusters: C{int} -> The number of Clusters.
        @param batch_size: C{int} -> The Batch size.
        @param filenames: C{list} -> The Filenames for the data.
        @param kwargs: C{dict} -> For futre use.
        @local config: C{config.Config} -> the config & hooks handler for the estimator.
        """
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.filenames = filenames
        self.config = KmeansConfig(self.base_model_dir)

    def run(self):
        """
        @return C{MiniBatchKmeansWraperOutput} -> the clustering result, used for IOWraper.
        @remarks:
                 *This is where the action starts.
        """     
        self.cluster = tf.compat.v1.estimator.experimental.KMeans(
            num_clusters=self.num_clusters,
            use_mini_batch=False,
            config=self.config.get_run_config(),
        ) 

        self._train()
        self.wraper_output = self._transform()
        if self._use_tensorboard:
            self._tensorboard(batch_size=self.batch_size)
        
        return self.wraper_output

    def _train(self):
        """
        @remarks:
                 *Trains the estimator.
        """
        with tf.device('/device:GPU:0'):
            logging.info("Starting to train")
            self.cluster.train(input_fn=lambda: self._input_fn(                                
                                    batch_size=self.batch_size,
                                    shuffle=False, 
                                    num_epochs=self.num_epochs,
                                    filenames=self.filenames))
                                    
            self.score = self.cluster.score(input_fn=lambda: self._input_fn(                                       
                                        batch_size=self.batch_size,
                                        shuffle=False, 
                                        num_epochs=self.num_epochs,
                                        filenames=self.filenames))

            logging.info(f"score: {self.score}")
    
    def _transform(self):
        """
        @return C{MiniBatchKmeansWraperOutput}  -> The clustering output, used for IOWraper.
        @remarks:
                 *splits the input data to each cluster.
        """
        with tf.device('/device:gpu:0'):
            # ?map the input points to their clusters
            logging.info("starting to Transform the data")
            cluster_indices = list(self.cluster.predict_cluster_index(input_fn=lambda: self._input_fn(                                                                  
                                                                    batch_size=self.batch_size,
                                                                    shuffle=False, 
                                                                    num_epochs=self.num_epochs,
                                                                    filenames=self.filenames)))
        logging.debug(f"There are {len(cluster_indices)} labels")       
        return MiniBatchKmeansWraperOutput(cluster_labels=cluster_indices)
    
class MiniBatchKmeansWraperOutput(WraperOutput):
    """MiniBatchKmeansWraperOutput -> A WraperOutput Object for MiniBatchKemansTensorflowWraper."""
    def __init__(self, cluster_labels):
        n_clusters = max(cluster_labels) + 1
        super().__init__(n_clusters, cluster_labels)
