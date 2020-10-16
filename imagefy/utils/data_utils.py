"""
@Author: Yuval Kaneti 
"""

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import logging; logger = logging.getLogger('Imagefy')
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from imagefy.utils.web_utils import warning_cleanup
from imagefy.utils.common import INCEPTION_RESNET_TENSORFLOW_WRAPER_OUTPUT, INCEPTION_RESNET_IMAGE_SIZE

class Image():
    """Image -> A Class That holds information about an Image."""
    def __init__(self, file_name: str, data: np.ndarray):
        """
        @param file_name: C{str} -> The name of the image
        @param data: C{np.array} -> A np.array containing the image data
        """
        self.file_name = file_name
        self.data = data
        self.shape = data.shape
        self.cluster_n = None
        self.top = False
        self.score = None

    def __str__(self):
        return f"filename: {self.file_name}, cluster: {self.cluster_n}, top: {self.top}, shape: {self.shape}, score: {self.score}"

    def free(self):
        """
        @remarks *Deletes the image data from memory.
                 *Should be Tested to make sure it actually does somthing.
        """
        logger.debug(f"Freeing {sys.getsizeof(self.data)} bytes from {self.basename}")
        del self.data
        self.data = None

class WraperOutput(ABC):
    """WraperOutput -> An Abstarct Class represnting a BaseWraper Return Value."""
    def __init__(self, n_clusters: int, cluster_labels: list):
        """
        @param n_cluster: C{int} -> The nmber of clusters.
        @param cluster_labels: C{list} -> the corosponding cluster index ordered by the data originial order.
        @remarks *DO NOT USE THIS CLASS, inherite From it.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.cluster_labels = cluster_labels
        self.name = self.__class__.__name__

class BaseLoader(ABC):
    """BaseLoader -> A BaseLoader Abstract class."""
    def __init__(self, images: list, image_size: int, **kwrags):
        """
        @param images: C{list} -> A list of ByteIO Fd from streamlit.file_uploader
        @param image_size>: C{int} -> The image size to resize to.
        @param kwargs: C{dict} -> for future meeds.
        @local name: C{list} -> The name of the Class. Parents included.
        """
        self.name = self.__class__.__name__
        logger.debug(f"Initializing {self.name}")
        self.images = images
        self.image_size = image_size
        self.data = self.preprocess_images()

    @abstractmethod
    def run(self):
        """
        @remarks *Legecy Fucntion for suported DataLoader.
        """
        logger.info(f"Starting {self.name}")
    
    def preprocess_images(self):
        """
        """
        data = []
        valid_images = []
        logger.info("Preprocessing Data")
        for image_fd in self.images:
            if self.valid_data(image_fd):
                valid_images.append(image_fd)
                data.append(np.array(image_fd.read()))
           
        self.images = valid_images
        logger.info("Finished Preprocessing Data")
        if len(data) < 1:
            warning_cleanup(f"Please Press The Start button again, this is a Known issue")

        return np.array(data)

    def valid_data(self, image_fd):
        first_byte = image_fd.read(1)
        if first_byte != b'':
            image_fd.seek(0)
            return True
        else:
            image_fd.seek(0)
            return False

    def flush(self):
        del self.data
        self.data = None

class TensorLoader(BaseLoader):
    """TensorLoader -> A Scalbles solution for loading data using tf.data.Dataset API."""
    def __init__(self, model_name: str, base_path: str, **kwargs : dict):
        """
        @param model_name: C{str} -> The model name, Taken for BaseSuit.model_name.
        @param base_path: C{str} -> The base path for logs & output, Taken from BaseSuit.base_path.
        @param kwargs: C{dict} -> Other parameters for BaseLoader.
        @local AUTOTUNE C{tf.data.experimental.AUTOTUNE} -> An object that can mange resources at runtime
        @local dataset: C{tf.data.Dataset} -> The Dataset created with either *_input_fn.
        @local _image_names: C{list} -> A list of all the images in the self.dir_path, used for tensorboard logs.
        @local output_dir_path: C{str} -> The output path of IOWraper, used for InceptionResnetWraper input_fn.
        """
        super().__init__(**kwargs)
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.dataset = None
        self._image_names = np.asarray([image.name for image in self.images])
        self.batch_size = len(self._image_names) if len(self._image_names) < 80 else len(self._image_names) // 2
        self.num_epochs = 1

        # del self.images
        self.model_name = model_name
        self.base_path = base_path
   
    def kmeans_input_fn(self, **kwrags):
        """
        @param batch_size: C{int} -> The batch size for tf.data.Dataset.batch(...).
        @param shuffle: C{bool} -> Whether to shuffle the Dataset (during .list_files for better preformence).
        @param num_epochs: C{int} -> The number of epochs for tf.data.Dataset.repeat(...).
        @param kwargs: C{dict} -> For future needs.
        @return tf.data.Dataset -> the dataset for KmeansTensorflowWraper
        """
        options = tf.data.Options()
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = True
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data)
        self.dataset = self.dataset.map(self._preprocess_tensor, num_parallel_calls=self.AUTOTUNE)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True).repeat(self.num_epochs)
        
        self.dataset = self.dataset.cache()
        # self.dataset = self.dataset.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0', buffer_size=self.AUTOTUNE))
        self.dataset = self.dataset.apply(tf.data.experimental.copy_to_device("/device:gpu:0"))
        self.dataset = self.dataset.prefetch(self.AUTOTUNE)
        self.dataset = self.dataset.with_options(options)

        return self.dataset   
 
    def inception_input_fn(self):
        """
        @return tf.data.Dataset -> the dataset for Inception
        """
        options = tf.data.Options()
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = True
        dataset = tf.data.Dataset.from_tensor_slices(self.data)
        dataset = dataset.map(self._preprocess_tensor, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTOTUNE)
        dataset = dataset.with_options(options)
        return dataset
        
    @tf.function
    def _preprocess_tensor(self, image_rawe: str):
        """
        @param image_rawe: C{tf.Tensor} -> The image raw form, created by streamlit.file_uploader & self.preprocess_images(...).
        @return C{tf.Tensor} -> A tensor image, normalized (-1, 1) & flattended.
        """   
        image = tf.image.decode_jpeg(image_rawe)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[INCEPTION_RESNET_IMAGE_SIZE, INCEPTION_RESNET_IMAGE_SIZE])
        image = image - tf.math.reduce_mean(input_tensor=image, axis=0)
        image = image / tf.math.reduce_max(input_tensor=tf.abs(image), axis=0)
        image = tf.reshape(image, [-1])
        return image

    def run(self, **kwrags):
        """
        @param kwargs: C{dict} -> for future needs.
        @remarks *A simple function the will list all the supported images in a folder.
                 *The list of supported image types can be found at @common.IMAGE_TYPES.
        @return C{list} -> A list of `hollow` Image Objects for IOWraper with KmeansTensorflowWraper
        """
        logger.debug("Listing Images")
        image_list = []
        for image_name, data in zip(self._image_names, self.data):
            image_list.append(Image(image_name, data=data))

        # Delete self.data, its not neaded anymore.
        # self.flush()
        return image_list

class IOWraper():
    """IOWraper -> An Object which handles multithreaded IO Opartions."""
    def __init__(self, images: list, wraper_output: WraperOutput, model_name: str, base_path: str, **kwargs: dict):
        """
        @param images: C{list} -> A List of Image Objects.
        @param wraper_output: C{WraperOutput} -> The Output returned by on of the @BaseWrapers.
        @remarks *@self.wraper_output can be any of the inherited object from WraperOutput.
                 *@self._clean is invoked on __init__, it will remove all the old cluster output directories.    
        """
        self.images = images
        self.inception_data = None
        self.wraper_output = wraper_output
        self.model_name = model_name
        self.base_path = base_path

    def reset_cluster_labels(self):
        logger.debug("Reseting Cluster Labels")
        for image in self.images:
            image.cluster_n = 0

    def set_inception_data(self, wraper_output: WraperOutput):
        """
        @param wraper_output: C{WraperOutput} -> The result of InceptionResnetTensorflowWraper.run(...).
        @remarks *This function is neaded for the second stage of Imagefy.
        """
        self.wraper_output = wraper_output

    def merge_inception_data(self, top: int):
        """
        @param top: C{int} -> The top number of images to move from each cluster.
        @remarks *This function is neaded for the second stage of Imagefy.
        """
        if self.wraper_output.name != INCEPTION_RESNET_TENSORFLOW_WRAPER_OUTPUT:
            raise RuntimeError("wraper_output is the kmeans one, use IOWraper.set_inception_data(...) to set this varibles")

        logger.debug("Merging Inception data & WraperOutput")
        for (image, predictor_output) in zip(self.images, self.wraper_output.top(k=top)):
            image.cluster_n = int(predictor_output.label)
            image.score = predictor_output.score
            if predictor_output.top:
                image.top = True

        return self.images
