#! /usr/bin/env python3
"""
Code Taken From: https://github.com/tensorflow/tensorflow/issues/20942 @kevintrankt commented on Jul 23, 2018
Code Taken from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
Wiki: Silhouette clustering -> https://en.wikipedia.org/wiki/Silhouette_(clustering)
Author: Yuval Kaneti⭐
"""

#### Imports ####
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from tqdm import tqdm
import logging 
from pprint import pformat
import tensorflow as tf; tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from integration.wrapers.base_wraper import BaseWraper
from integration.utils.data_utils import WraperOutput, DataGenerator

class MiniBatchKmeansTensorflowWraper(BaseWraper):
    """MiniBatchKmeansTensorflowWraper -> An implemntion of Minibatch Kmeans in Tensorflow."""
    def __init__(self, data : list, num_iterations : int, num_clusters : int, save : bool, **kwargs : dict):
        """
        @param data: C{list} -> a list of Image Objects.
        @param num_iterations: C{int} -> The number of Training iterations.
        @param num_clusters: C{int} -> The number of Clusters.
        @param save: C{bool} -> Should we Save the Trained Model, NO IMPLEMNTED YET.
        """
        super().__init__()#**kwargs
        logging.debug("Initialzing MiniBatchKmeansTensorflowWraper")

        self._data = data
        self.num_iterations = num_iterations
        self.num_clusters = num_clusters 
        self._save = save if not save else False; logging.warn(f"{self.name}.Save() is not implemnted yet.`")
        # self._load = load ..
        self.cluster_centers = None

    def _input_fn(self):
        (self.X, self.y) = self.Wraperize()
        return tf.compat.v1.train.limit_epochs(
            tf.convert_to_tensor(self.X, dtype=tf.float32), num_epochs=1)

    def run(self):
        """
        @remarks *This is where the action starts.
        """
        # (self.X, self.y) = self.Wraperize()
        config = tf.compat.v1.estimator.RunConfig(
            model_dir=self.model_dir,
            save_summary_steps=100,
            keep_checkpoint_max=1
        )
        self.cluster = tf.compat.v1.estimator.experimental.KMeans(
            num_clusters=self.num_clusters,
            use_mini_batch=True,
            config=config)

        self._train()
        transformed = self._transform()
        # if self.save:
        #     self.save()

        return transformed
    def _train(self):
        # train
        logging.info("Starting to train")
        previous_centers = None
        for i in tqdm(range(self.num_iterations)):
            # tqdm.write(f"Running iteration {i}")
            self.cluster.train(self._input_fn)
            self.cluster_centers = self.cluster.cluster_centers()
            if previous_centers is not None:
                pass
                # logging.debug(f"delta: {pformat(self.cluster_centers - previous_centers)}")
            previous_centers = self.cluster_centers
            tqdm.write(f"score: {self.cluster.score(self._input_fn)}")
            # logging.info (f"cluster centers: {pformat(self.cluster_centers)}")

    def _transform(self):
        # map the input points to their clusters
        logging.info("starting to Transform the data")
        cluster_indices = list(self.cluster.predict_cluster_index(self._input_fn))
        labels = []
        for i, point in enumerate(self.X):
            cluster_index = cluster_indices[i]
            center = self.cluster_centers[cluster_index]
            logging.debug(f"point: {point} is in cluster: {cluster_index} centered at: {center}")
            labels.append(cluster_index)

        return MiniBatchKmeansWraperOutput(cluster_labels=labels)

    def save(self):
        # Saving estimator model
        model_path = os.path.join(self.model_dir, self.model_name)
        logging.info(f"Saving model to {model_path}")
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(self.input_fn())
        serving_input_fn = tf.compat.v1.estimator.export.build_parsing_serving_input_receiver_fn(
        tf.feature_column.make_parse_example_spec(feature_columns))
        self.cluster.estimator.export_saved_model(model_path, serving_input_fn)
    
    def load(self, path):
        logging.info(f"loading model from {path}")
        self.cluster = tf.saved_model.load(path)

class MiniBatchKmeansWraperOutput(WraperOutput):
    """MiniBatchKmeansWraperOutput -> A WraperOutput Object for MiniBatchKemansTensorflowWraper."""
    def __init__(self, cluster_labels):
        n_clusters = max(cluster_labels) + 1
        logging.warn(f"there are {n_clusters} clusters?")
        super().__init__(n_clusters, cluster_labels)
    