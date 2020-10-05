
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from integration.utils.data_utils import BaseLoader, TensorLoader
from integration.utils.tensorboard_utils import save_embeddings, save_labels, save_sprite_image


class TensorboardWraper(BaseLoader):
    def __init__(self, name: str, base_model_dir: str, y: list, batch_size: int, data_length: int, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.name = name
        self.base_model_dir = base_model_dir
        self.y = y
        self.metadata_path = os.path.join(self.base_model_dir, f"{self.name}-metadata.tsv")
        self._loader = TensorLoader(**self.kwargs)
        self.batch_size = batch_size
        self.dataset = self._loader.run(batch_size=None, shuffle=False, num_epochs=1)
        self.data_length = data_length
        self.embedings = []
        self.images = []

    def run(self):
        # iterator = self.dataset.make_one_shot_iterator()
        iterator = tf.compat.v1.data.make_one_shot_iterator(self.dataset)

        next_element = iterator.get_next()
        with tf.compat.v1.Session() as sess:
            try:
                logging.info("Starting to load data")
                while True:
                    output = sess.run(next_element)
                    # for image in [*output.reshape((self.batch_size, self.image_size * self.image_size * 3))]:
                    #     self.images.append(np.reshape(image,(self.image_size, self.image_size, 3)))
                    #     self.embedings.append(tf.Variable(image, dtype=tf.float32))
                    #     logging.debug("for loop")

                    # logging.debug("iterating data")
                    self.images.append(output.reshape((self.image_size, self.image_size, 3)))
                    self.embedings.append(tf.Variable(output, dtype=tf.float32))   
                      
            except tf.errors.OutOfRangeError:
                logging.info("Finished Loading Data")
            
            #DEBUG
            logging.debug(f"There are {len(self.images)} images")
            logging.debug(f"The are {len(self.embedings)} embedings")
            logging.debug(f"Truncating dataset to {self.data_length}")
            self.images = self.images[:self.data_length]
            self.embedings = self.embedings[:self.data_length]
            self.embedings = tf.stack(self.embedings)

            self.writer = tf.compat.v1.summary.FileWriter(self.base_model_dir, graph=None)
            self.projector_config = projector.ProjectorConfig()
            self.embedding_config = self.projector_config.embeddings.add()

            self.log_embedings()
            self.log_sprite()
            # self.log_histogram()
            # self.log_images()

            logging.debug(f"Saving Projector Embeddings")
            projector.visualize_embeddings(self.writer, self.projector_config)

            saver = tf.compat.v1.train.Saver(max_to_keep=1)
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, os.path.join(self.base_model_dir, 'ckpt'))


    def log_embedings(self):
            embedding_var = tf.Variable(self.embedings, name='embedings', trainable=False)
            self.embedding_config.tensor_name = embedding_var.name
            self.embedding_config.metadata_path = self.metadata_path

    def log_sprite(self):
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
        hist = tf.HistogramProto()
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
        logging.debug(f"Saving labels to {self.base_model_dir}")
        save_labels(self.y, self.name, self.base_model_dir)

class Tensorboard():
    """Tensorboard -> A Class that will generate Tensorboard projector Files."""
    def __init__(self, data: list, X: np.ndarray, y: list, name: str, base_model_dir: dir, image_size: int, **kwrags):
        """
        @param data: C{list} -> A List of Image Objects.
        @param X: Optional C{np.ndarray} -> A np.ndarray of Image.data Arrrays.
        @param y: Optional C{list} -> A list of Image.n_cluster, Could be improved into cluster_{cluster_n}
        @remarks: *Normal usage will provide the data varibale and X,y will be extracted via self._convert_image_object_to_numpy,
                       In the Future Image object itself will have this builtin.
                  *When Using Tensorboard.load(), on initiliazition, specify data=None, a warning will be logged,
                       It will remind you to use Tensorboard.load() because no data is loaded.
        """
        self.data = data
        self.X = X
        self.y = y
        self.name = name
        self.base_model_dir = base_model_dir
        self.image_size = image_size

        if self.data is not None and self.data[0].data is not None:
            self._convert_image_object_to_numpy()
        elif self.X is not None and self.y is not None:
            logging.debug("X, y were Found")
            logging.debug(f"X: {self.X.shape}")
            logging.debug(f"Y: ({len(self.y)}, 1)")
        else:
            logging.warn(f"Please Make sure to use Tensorboard.load() OR you called Image.free(), Data is {type(self.data)}")

    def _convert_image_object_to_numpy(self):
        """
        @param data: C{list} -> a list on Image Objects.
        @remarks *Will Split an Image object into Image.data & image.cluster_n (aka X, y).
                *Make sure you call @IOWraper.marge_data() befor Visualizing.
                *RuntimeWarning will be raised if there are missing self.embedings.
                *Will be implemnted into the Image object in the future.
        """
        logging.debug("Transforming data to visualization format")
        self.X = np.asarray([image.data for image in self.data])
        self.y = [image.cluster_n for image in self.data if image.cluster_n is not None]
        if len(self.y) == 0:
            raise RuntimeWarning("Make sure you @IOWraper.marge_data(), no Labels are Avilable")

    def run(self):
        """
        @param name: C{str} -> The Tensor name, doesnt really have a meaning in this context.
        @remarks *Might launch a subprocess that will run Tensorboard & open Chrome in the futre
        """
        if self.X is not None and self.y is not None:
            logging.info("Creating Tensorboard metadata")
            images_features_labels = {}
            images = np.asarray([arr.reshape((self.image_size, self.image_size, 3)) for arr in self.X])
            images_features_labels[self.name] = [images, self.X, self.y]
            logging.info("Saving Embeddings")
            # log_name = f"{self.name}-{time.time()}"
            tensorboard_path = os.path.join(self.base_model_dir) # TENSORBOARD_LOG_DIR # ,log_name
            save_embeddings(images_features_labels, tensorboard_path)
            logging.info(f"Run tensorboard --logdir={tensorboard_path}")
            logging.info("Go to http://localhost:6006/ and click Projector")
        else:
            logging.warn(f"Please Make sure to use TensorboardWraper.load() OR you called Image.free(), Data is {type(self.data)}")
