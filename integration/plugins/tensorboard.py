
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from integration.utils.data_utils import BaseLoader, TensorLoader
from integration.utils.tensorboard_utils import save_embeddings, save_labels, save_sprite_image


class TensorboardWraper(BaseLoader):
    def __init__(self, name: str, base_model_dir: str, y: list, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.name = name
        self.base_model_dir = base_model_dir
        self.y = y
        self.metadata_path = os.path.join(self.base_model_dir, f"{self.name}-metadata.tsv")
        self._loader = TensorLoader(**self.kwargs)
        self.dataset = self._loader.run(batch_size=None, shuffle=False, num_epochs=1)

    def run(self):
        # iterator = self.dataset.make_one_shot_iterator()
        iterator = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        embedings = []
        images = []
        next_element = iterator.get_next()
        with tf.compat.v1.Session() as sess:
            try:
                logging.info("Starting to load data")
                while True:
                    output = sess.run(next_element)
                    images.append(output.reshape((self.image_size, self.image_size, 3)))
                    embedings.append(tf.Variable(output, dtype=tf.float32))     
            except tf.errors.OutOfRangeError:
                logging.info("Finished Loading Data")

            embedings = tf.stack(embedings)
            embedding_var = tf.Variable(embedings, name='embedings', trainable=False)
            writer = tf.compat.v1.summary.FileWriter(self.base_model_dir, graph=None)
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = self.metadata_path

            image_filename = f"embedings.png"
            sprites_path = os.path.join(self.base_model_dir, image_filename)
            logging.debug(f"Saving images to {image_filename}")
            images = np.asarray(images)
            save_sprite_image(images, path=sprites_path, invert=False)#len(images.shape)<4)
            embedding.sprite.image_path = sprites_path
            embedding.sprite.single_image_dim.extend(images[0].shape)
    
            logging.debug(f"Saving Projector Embeddings")
            projector.visualize_embeddings(writer, config)

            saver = tf.compat.v1.train.Saver(max_to_keep=1)
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, os.path.join(self.base_model_dir, 'ckpt'))

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
                *RuntimeWarning will be raised if there are missing values.
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
            log_name = f"{self.name}-{time.time()}"
            tensorboard_path = os.path.join(self.base_model_dir) # TENSORBOARD_LOG_DIR # ,log_name
            save_embeddings(images_features_labels, tensorboard_path)
            logging.info(f"Run tensorboard --logdir={tensorboard_path}")
            logging.info("Go to http://localhost:6006/ and click Projector")
        else:
            logging.warn(f"Please Make sure to use TensorboardWraper.load() OR you called Image.free(), Data is {type(self.data)}")
