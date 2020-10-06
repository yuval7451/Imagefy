#! /usr/bin/env python3
#Author: Yuval Kaneti⭐

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import glob
import shutil
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import concurrent.futures
from abc import ABC, abstractmethod
from integration.utils.common import OUTPUT_DIR_PATH, CLUSTER_NAME_FORMAT, MAX_WORKERS

class Image():
    """Image -> A Class That holds information about an Image."""
    def __init__(self, src_path : str, data : list, hollow : bool=False):
        """
        @param src_path: C{str} -> The src path of the image
        @param data: C{list} -> A np.array.flatten() containing the image data
        """
        self.src_path = src_path
        self.basename = os.path.basename(src_path)
        self.dst_dir = None
        self.dst_path = None
        self.cluster_n = None
        self.hollow = hollow
        self.data = data
        self.top = False
        if not hollow:
            self.shape = self.data.shape
        else:
            self.shape = (None,)
    
    def __str__(self):
        return f"src: {self.src_path}, dst: {self.dst_path}, cluster: {self.cluster_n}, hollow: {self.hollow}, top {self.top}"

    def free(self):
        """
        @remarks *Deletes the image data from memory.
                 *Should be Tested to make sure it actually does somthing.
        """
        logging.debug(f"Freeing {sys.getsizeof(self.data)} bytes from {self.basename}")
        del self.data
        self.data = None

    def flush(self):
        """
        @remarks *Only if self.dst_dir was chosen the dst_path can be filled.
                 *Can only be used after IOWraper.merge_data() was invoked.
        """
        if self.dst_dir:
            self.dst_path = os.path.join(self.dst_dir, self.basename)
        else:
            logging.error("Call IOWraper.merge_data() First")

        #     x = np.asarray([ImageObj.data for ImageObj in self._data])

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

class BaseLoader(ABC):
    """BaseLoader -> A BaseLoader Abstract class."""
    def __init__(self, dir_path: str, image_size: int, **kwrags):
        """
        @param dir_path: C{str} -> The dir path to load images from.
        @param image_size>: C{int} -> The image size to resize to.
        @param kwargs: C{dict} -> for future meeds.
        @local name: C{list} -> The name of the Class. Parents included.
        """
        self.dir_path = dir_path
        self.image_size = image_size
        self.name = self.__class__.__name__
        logging.debug(f"Initializing {self.name}")

    @abstractmethod
    def run(self):
        """
        @remarks *Legecy Fucntion for suported DataLoader.
        """
        logging.info(f"Starting {self.name}")
    
class TensorLoader(BaseLoader):
    """TensorLoader -> A Scalbles solution for loading data using tf.data.Dataset API."""
    def __init__(self, model_name: str, base_path: str, output_dir_path: str=None, **kwargs : dict):
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
        self._image_names = np.asarray([image_name for image_name in glob.glob(self.dir_path)])
        self.model_name = model_name
        self.base_path = base_path
        if output_dir_path is None:
            self.output_dir_path = self.dir_path #os.path.join(self.base_path, OUTPUT_DIR_PATH, self.model_name)
        else:
            self.output_dir_path = output_dir_path

    @tf.function
    def mini_batch_kmeans_input_fn(self, batch_size: int, shuffle: bool, num_epochs: int, **kwrags):
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
        self.dataset = tf.data.Dataset.list_files(self.dir_path, shuffle=shuffle)
        self.dataset = self.dataset.map(self._load_tensor, num_parallel_calls=self.AUTOTUNE).repeat(num_epochs)
        if batch_size is not None:
            self.dataset = self.dataset.batch(batch_size, drop_remainder=True).cache() #.prefetch(self.AUTOTUNE)
        else:
            self.dataset = self.dataset.cache() #.prefetch(self.AUTOTUNE)
        self.dataset = self.dataset.apply((tf.data.experimental.prefetch_to_device('/device:GPU:0', buffer_size=self.AUTOTUNE)))
        self.dataset = self.dataset.with_options(options)
        return self.dataset   
 
    @tf.function
    def inception_input_fn(self):
        """
        @return tf.data.Dataset -> the dataset for Inception
        """
        options = tf.data.Options()
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = True
        dataset = tf.data.Dataset.list_files(os.path.join(self.output_dir_path, "*"), shuffle=False)
        dataset = dataset.map(self._load_tensor_with_label, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.with_options(options)
        dataset = dataset.prefetch(self.AUTOTUNE)#.cache()   
        return dataset

    @tf.function
    def _load_tensor(self, image_path: str):
        """
        @param image_path: C{tf.Tensor} -> The image path Tensor, created by tf.data.Dataset.list_files(...).
        @return C{tf.Tensor} -> A tensor image, normalized (-1, 1) & flattended.
        """   
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[self.image_size, self.image_size])
        image = image - tf.math.reduce_mean(image, axis=0)
        image = image / tf.math.reduce_max(tf.abs(image), axis=0)
        image = tf.reshape(image, [-1])
        return image

    @tf.function
    def _load_tensor_with_label(self, image_path: str):
        """
        @param image_path: C{tf.Tensor} -> The image path Tensor, created by tf.data.Dataset.list_files(...).
        # @return C{tf.Tensor} -> A tensor image, normalized (-1, 1) & flattended.
        """   
        image = tf.io.read_file(image_path)
        image_path_parts = tf.strings.split(image_path, os.sep, result_type='RaggedTensor')
        image_name = image_path_parts[-1]
        label = image_path_parts[-2]
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[224, 224])
        image = image - tf.math.reduce_mean(image, axis=0)
        image = image / tf.math.reduce_max(tf.abs(image), axis=0)
        image = tf.reshape(image, [-1])
        return image, label, image_name

    def run(self, **kwrags):
        """
        @param kwargs: C{dict} -> for future needs.
        @remarks *A simple function the will list all the supported images in a folder.
                 *The list of supported image types can be found at @common.IMAGE_TYPES.
        @return C{list} -> A list of `hollow` Image Objects for IOWraper with KmeansTensorflowWraper
        """
        logging.debug("Listing Images")
        image_list = []
        for image_name in self._image_names:
            image_path = os.path.join(self.dir_path, image_name) 
            image_list.append(Image(src_path=image_path, data=None, hollow=True))

        logging.debug(f"{len(image_list)} hollow images")
        return image_list

class IOWraper():
    """IOWraper -> An Object which handles multithreaded IO Opartions."""
    def __init__(self, data: list, wraper_output: WraperOutput, model_name: str, base_path: str, **kwargs: dict):
        """
        @param data: C{list} -> A List of Image Objects.
        @param wraper_output: C{WraperOutput} -> The Output returned by on of the @BaseWrapers.
        @remarks *@self.wraper_output can be any of the inherited object from WraperOutput.
                 *@self._clean is invoked on __init__, it will remove all the old cluster output directories.    
        """
        self.data = data
        self.wraper_output = wraper_output
        self.model_name = model_name
        self.base_path = base_path
        # self._clean() DONT USE

    def _clean(self):
        """
        @remarks *Will remove all the old cluster output directories at @OUTPUT_DIR_PATH.
                 *Using shutil.rmtree is dangerous, Take a Good look at @OUTPUT_DIR_PATH.
        """
        logging.debug("Cleaning Up from last run")
        for _dir in os.listdir(OUTPUT_DIR_PATH):
           shutil.rmtree(os.path.join(OUTPUT_DIR_PATH, _dir))

    def merge_data(self):
        """
        @remarks *will use the data from self.wraper_output (@WraperOutput object) to set the Image.dst_fir & Image.cluster_n.
                 *This function must be called befor @IOWraper.move_data().
                 *A usual use case will look like this:
                  IOWraper.create_output_dirs()
                  IOWraper.merge_data()
                  IOWraper.move_data()
        """
        logging.debug("Merging Images & WraperOutput")
        for (image, cluster_label) in zip(self.data, self.wraper_output.cluster_labels):
            image.dst_dir = os.path.join(self.base_path, OUTPUT_DIR_PATH, self.model_name, CLUSTER_NAME_FORMAT.format(cluster_label))
            image.cluster_n = int(cluster_label)
            # image.free() -> Only free if You dont want to use Tensorboard.
            image.flush()
            # logging.debug(image)

    def create_output_dirs(self):
        """
        @remarks *A simple Function that will Create multiple directories using multiple Workers.
                 *ThreadPoolExecutor(@MAX_WORKERS).
                 *Might not be neaded for a small amount of directories.
                 *@IOWraper.move_data() can't be invoked befor this function, not output directories will exist.
                 *If an error acourd it will be loged & raised.
        """
        logging.debug("Creating Output Directories")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_dir = {}
            for dir_index in range(self.wraper_output.n_clusters):
                dir_path = os.path.join(self.base_path, OUTPUT_DIR_PATH, self.model_name, f"cluster_{dir_index}")
                future_to_dir[executor.submit(os.makedirs, dir_path)] = dir_index

            dir_path = os.path.join(self.base_path, OUTPUT_DIR_PATH, self.model_name, "output")
            future_to_dir[executor.submit(os.makedirs, dir_path)] = None 

            for future in concurrent.futures.as_completed(future_to_dir):
                    result = future.result()

        logging.debug("Finished Creating Output Directories")

    def move_data(self):
        """
        @remarks *Will Copy The Images from Image.src_path to Image.dst_path Using shutil & ThreadPoolExecutor(@MAX_WORKERS)
                 *@IOWraper.merge_data MUST be invoked befor this function or a RuntimeError will be raised.
                 *Once shutil.copy is invoked with image.dst_path = None. No Warning is given at this stage.
        """
        logging.debug("Moving Data")
        with tqdm(total=len(self.data)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_file_moved = {executor.submit(shutil.copy, image.src_path, image.dst_path): image for image in self.data if image.src_path is not None and image.dst_path is not None}
                for future in concurrent.futures.as_completed(future_to_file_moved):
                        result = future.result()
                        pbar.update(1)
                        
        logging.debug("Finished Moving data")

class TensorIOWraper():
    """TensorIOWraper -> An Object which handles multithreaded Tensor IO Opartions."""
    def __init__(self, data: list, wraper_output: WraperOutput, model_name: str, base_path: str, **kwargs: dict):
        """
        @param data: C{list} -> A List of Image Objects.
        @param wraper_output: C{WraperOutput} -> The Output returned by on of the @BaseWrapers.
        @remarks *@self.wraper_output can be any of the inherited object from WraperOutput.
                 *@self._clean is invoked on __init__, it will remove all the old cluster output directories.    
        """
        self.data = data
        self.wraper_output = wraper_output
        self.model_name = model_name
        self.base_path = base_path
        # self._clean() DONT USE


    def merge_data(self):
        """
        @remarks *will use the data from self.wraper_output (@WraperOutput object) to set the Image.dst_fir & Image.cluster_n.
                 *This function must be called befor @IOWraper.move_data().
                 *A usual use case will look like this:
                  IOWraper.create_output_dirs()
                  IOWraper.merge_data()
                  IOWraper.move_data()
        """
        logging.debug("Merging Images & WraperOutput")
        for (image, cluster_label) in zip(self.data, self.wraper_output.cluster_labels):
            image.dst_dir = os.path.join(self.base_path, OUTPUT_DIR_PATH, self.model_name, CLUSTER_NAME_FORMAT.format(cluster_label))
            image.cluster_n = int(cluster_label)
            # image.free() -> Only free if You dont want to use Tensorboard.
            image.flush()
            # logging.debug(image)

    def create_output_dirs(self):
        """
        @remarks *A simple Function that will Create multiple directories using multiple Workers.
                 *ThreadPoolExecutor(@MAX_WORKERS).
                 *Might not be neaded for a small amount of directories.
                 *@IOWraper.move_data() can't be invoked befor this function, not output directories will exist.
                 *If an error acourd it will be loged & raised.
        """
        logging.debug("Creating Output Directories")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_dir = {}
            for dir_index in range(self.wraper_output.n_clusters):
                dir_path = os.path.join(self.base_path, OUTPUT_DIR_PATH, self.model_name, f"cluster_{dir_index}")
                future_to_dir[executor.submit(os.makedirs, dir_path)] = dir_index

            dir_path = os.path.join(self.base_path, OUTPUT_DIR_PATH, self.model_name, "output")
            future_to_dir[executor.submit(os.makedirs, dir_path)] = None 

            for future in concurrent.futures.as_completed(future_to_dir):
                    result = future.result()

        logging.debug("Finished Creating Output Directories")

    def move_data(self):
        """
        @remarks *Will Copy The Images from Image.src_path to Image.dst_path Using shutil & ThreadPoolExecutor(@MAX_WORKERS)
                 *@IOWraper.merge_data MUST be invoked befor this function or a RuntimeError will be raised.
                 *Once shutil.copy is invoked with image.dst_path = None. No Warning is given at this stage.
        """
        logging.debug("Moving Data")
        with tqdm(total=len(self.data)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_file_moved = {executor.submit(shutil.copy, image.src_path, image.dst_path): image for image in self.data if image.src_path is not None and image.dst_path is not None}
                for future in concurrent.futures.as_completed(future_to_file_moved):
                        result = future.result()
                        pbar.update(1)
                        
        logging.debug("Finished Moving data")

