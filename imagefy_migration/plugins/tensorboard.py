# https://github.com/tensorflow/tensorboard/issues/2471

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from imagefy_migration.utils.data_utils import BaseLoader
from imagefy_migration.utils.common import EMBEDDINGS_TENSOR_NAME
from imagefy_migration.utils.tensorboard_utils import save_labels, save_sprite_image


class TensorboardWraper(BaseLoader):
    """TensorboardWraper -> A class that will log tensorboard format output for projection & Graphs."""
    def __init__(self, name: str, base_model_dir: str, metadata: tuple, batch_size: int, data_length: int, dataset: callable, **kwargs):
        """
        @param  name: C{str} -> The name of the Wraper the called TensorboardWraper, Take from BaseSuit.name.
        @param  base_model_dir: C{str} -> The base model dir for output & logs, Take from BaseSuit.base_model_name.
        @param  metadata: C{list} -> A list containing (labels, filenames) -> (WraperOutput.cluster_labels, BaseLoader._image_names).
        @param  batch_size: C{int} -> The batch size for the TensorLoader input_fn.
        @param  data_length: C{int} -> The data_lenght, use for debuging, sometimes Labels and filenames dont match in length.
        @param  dataset: C{callable} -> The Wraper Tensorloader input_fn
        @param  **kwargs: C{dict} -> For future use.
        @local metadata_path: C{str} -> The path for embedings labels.
        @local embedings: C{list} -> A place holder of the embedings
        @local images: C{list} > A place holder of the images.
        """
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.name = name
        self.base_model_dir = base_model_dir
        self.metadata = metadata
        self.metadata_path = os.path.join(self.base_model_dir, f"{self.name}-metadata.tsv")
        self.batch_size = batch_size
        self.dataset =  dataset(batch_size=None, shuffle=False, num_epochs=1)
        self.data_length = data_length
        self.embedings = []
        self.images = []

    def run(self):
        """
        @remarks *Creates an iterator from The dataset input_fn.
                    *Dumps all the tesorboard logs to self.base_model_dir
        """
        logging.info("Starting to load data")
        for output in self.dataset:
            output_np = output.numpy()
            self.images.append(output_np.reshape((self.image_size, self.image_size, 3)))
            self.embedings.append(np.array(output_np, dtype=np.float32))   

        logging.info("Finished Loading Data")
        
        #DEBUG
        logging.debug(f"Truncating dataset to {self.data_length}")
        self.images = self.images[:self.data_length]
        self.embedings = np.array(self.embedings[:self.data_length])
        self.projector_config = projector.ProjectorConfig()
        self.embedding_config = self.projector_config.embeddings.add()

        self.log_embedings()
        self.log_sprite()
    
        logging.debug(f"Saving Projector Embeddings")
        projector.visualize_embeddings(self.base_model_dir, self.projector_config)

    def log_embedings(self):
        """
        """
        checkpoint_path = os.path.join(self.base_model_dir, 'embedings.ckpt')
        embedding_var = tf.Variable(self.embedings, name=EMBEDDINGS_TENSOR_NAME, trainable=False)
        ckpt = tf.train.Checkpoint(embeddings=embedding_var)
        ckpt.save(checkpoint_path)
        embeddings_tensor_name = EMBEDDINGS_TENSOR_NAME + "/.ATTRIBUTES/VARIABLE_VALUE"
        self.embedding_config.tensor_name = embeddings_tensor_name
        self.embedding_config.metadata_path = self.metadata_path

    def log_sprite(self):
        """
        """
        image_filename = f"embedings.png"
        sprites_path = os.path.join(self.base_model_dir, image_filename)
        logging.debug(f"Saving images to {image_filename}")
        self.images = np.asarray(self.images)
        save_sprite_image(self.images, path=sprites_path, invert=False) #len(images.shape)<4)
        self.embedding_config.sprite.image_path = sprites_path
        self.embedding_config.sprite.single_image_dim.extend(self.images[0].shape)
    
    def log_histogram(self, bins=1000):  
        # Create histogram using numpy
        counts, bin_edges = np.histogram(self.embedings, bins=bins)

        # Fill fields of histogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(self.embedings))
        hist.max = float(np.max(self.embedings))
        hist.num = int(np.prod(self.embedings.shape))
        hist.sum = float(np.sum(self.embedings))
        hist.sum_squares = float(np.sum(self.embedings**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]
        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="Histogram", histo=hist)])
        self.writer.add_summary(summary, 1)
        self.writer.flush()

    def log_images(self):
        image = tf.compat.v1.summary.image('Images', self.images, max_outputs=3)
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="Image", histo=image)])
        self.writer.add_summary(summary, 1)
        self.writer.flush()

    def save_labels(self):
        """
        """
        logging.debug(f"Saving labels to {self.base_model_dir}")
        save_labels(self.metadata, self.name, self.base_model_dir)
