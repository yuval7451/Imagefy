#! /usr/bin/env python3
#Author: Yuval Kaneti⭐

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import sys
import time
import glob
import shutil
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import concurrent.futures
from abc import ABC, abstractmethod
from integration.utils.common import IMAGE_TYPES, MAX_WORKERS, OUTPUT_DIR_PATH, CLUSTER_NAME_FORMAT

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
        if not hollow:
            self.shape = self.data.shape
        else:
            self.shape = (None,)
  
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
    def __init__(self, n_clusters : int, cluster_labels: list):
        """
        @param n_cluster: C{int} -> The nmber of clusters.
        @param cluster_labels: C{list} -> the corosponding cluster index ordered by the data originial order.
        @remarks *DO NOT USE THIS CLASS, inherite From it.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.cluster_labels = cluster_labels

class BaseLoader(ABC):
    def __init__(self, dir_path : str, image_size: int, **kwrags):
        self.dir_path = dir_path
        self.image_size = image_size
        self.data = None
        self.dtype = list
        self.name = self.__class__.__name__
        self._image_names = None

        logging.debug(f"Initializing {self.name}")

    @abstractmethod
    def run(self):
        logging.info(f"Starting {self.name}")
    
    # @abstractmethod
    # def tensorboard(self):
        # pass

class DataLoader(BaseLoader):
    """DataLoader -> a Threaded IO & Numpy Opertions Wraper."""
    def __init__(self, **kwargs : dict):
        """
        @param: dir_path: C{str} -> The dir to load the pictures from.
        @param: image_size: C{int} -> The size the images should be resized to befor flattening them [size, size, 3] -> [size * size * 3]
        """
        super().__init__(**kwargs)
        self.dataset = []
        # self.dtype = list
        self.dir_path = self.dir_path.split("*")[-2] #replace("*", "")
        self._image_names = None
        logging.debug(f"Dir path is: {self.dir_path}")


    def run(self):
        """
        @remarks *The only function that the user should call.
                 *Will start the invoketion of ThreadPoolExecutor(@MAX_WORKERS).
                 *There is an option to use ProcessPoolExecutor but the number of @MAX_WORKERS will have to be lower. 
                 *If an error acourd it will be loged & raised.
        """
        logging.info(f"Starting {self.name}")
        start = time.time()
        image_paths = self._list_images()

        logging.debug("Loading Images")
        with tqdm(total=len(image_paths)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_image = {executor.submit(self._load_image, image_path): image_path for image_path in image_paths}
                for future in concurrent.futures.as_completed(future_to_image):
                    data = future.result()
                    pbar.update(1)
                    self.dataset.append(data)
                    
        end = time.time() - start
        logging.info(f"It took {end} Seconds To Load {len(self.dataset)} Images")
        self._image_names = [image.basename for image in self.dataset]
        return self.dataset
        
    def _load_image(self, image_path : str):
        """
        @param image_path: C{str} -> The image path to load.
        @return: C{Image} -> An @Image Object containing information about the image.
        """
        # logging.debug(f"Loading {image_path}")
        np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), dtype=np.float32)
        # logging.debug(f"resizing {image_path}")
        np_image = np.resize(np_image, (self.image_size, self.image_size, 3))
        np_image = np_image.flatten()
        return Image(src_path=image_path, data=np_image)
        
    def _list_images(self):
        """
        @return C{list} -> A list of image paths
        @remarks *A simple function the will list all the supported images in a folder
                 *The list of supported image types can be found at @common.IMAGE_TYPES
        """
        logging.debug("Listing Images")
        image_list = []
        for image_name in os.listdir(self.dir_path):
            if image_name.split('.')[-1].lower() in IMAGE_TYPES:
                image_list.append(os.path.join(self.dir_path, image_name))
        return image_list

class TensorLoader(BaseLoader):
    """TensorLoader ."""
    def __init__(self, **kwargs : dict):
        """
        @param BaseLoader.dir_path: C{str} -> The dir to load the pictures from.
        @param BaseLoader.image_size: C{int} -> The size the images should be resized to befor flattening them [size, size, 3] -> [size * size * 3]
        @param AUTOTUNE C{tf.data.experimental.AUTOTUNE} -> ..
        @param tensor: C{bool} -> ..
        """
        super().__init__(**kwargs)
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.dtype = tf.data.Dataset
        self.dataset = None
        # self.tensorboard_dataset = self._tensorboard_dataset()
        self._image_names = [image_name for image_name in glob.glob(self.dir_path)]
        # self.iterator = None
    
    # @tf.function
    def run(self, batch_size: int, shuffle : bool, num_epochs : int, **kwrags):
        """
        """
        options = tf.data.Options()
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = True
        self.dataset = tf.data.Dataset.list_files(self.dir_path, shuffle=shuffle)   
        self.dataset = self.dataset.map(self._load_tensor, num_parallel_calls=self.AUTOTUNE).repeat(num_epochs)
        self.dataset = self.dataset.with_options(options)
        if batch_size is not None:
            self.dataset = self.dataset.batch(batch_size, drop_remainder=True).prefetch(self.AUTOTUNE).cache()
        else:
            self.dataset = self.dataset.prefetch(self.AUTOTUNE).cache()   
        # self.iterator = self.dataset.make_one_shot_iterator()
        return self.dataset
    
    # @tf.function
    def _load_tensor(self, image_path : str):
        """
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[self.image_size, self.image_size])
        image = tf.reshape(image, [-1])
        return image 

    def hollow_images(self):
        """
        @remarks *A simple function the will list all the supported images in a folder
                 *The list of supported image types can be found at @common.IMAGE_TYPES
        """
        logging.debug("Listing Images")
        image_list = []
        _dir_path = self.dir_path.replace("*", "")
        for image_name in os.listdir(_dir_path):
            if image_name.split('.')[-1].lower() in IMAGE_TYPES:
                image_path = os.path.join(_dir_path, image_name) 
                image_list.append(Image(src_path=image_path, data=None, hollow=True))

        return image_list

    # def _tensorboard_dataset(self):
    #     dataset = tf.data.Dataset.list_files(self.dir_path, shuffle=False)   
    #     dataset = dataset.map(self._load_tensor, num_parallel_calls=self.AUTOTUNE)
    #     dataset = dataset.cache().prefetch(self.AUTOTUNE)   
    #     return dataset

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
            # future_to_dir = {executor.submit(os.mkdir, f"cluster_{dir_index}"): dir_index for dir_index in range(self.silhouette_score.n_clusters)}
            future_to_dir = {}
            for dir_index in range(self.wraper_output.n_clusters):
                dir_path = os.path.join(self.base_path, OUTPUT_DIR_PATH, self.model_name, f"cluster_{dir_index}")
                future_to_dir[executor.submit(os.makedirs, dir_path)] = dir_index

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
                future_to_file_moved = {executor.submit(shutil.copy, image.src_path, image.dst_path): image for image in self.data}
                for future in concurrent.futures.as_completed(future_to_file_moved):
                        result = future.result()
                        pbar.update(1)
                        
        logging.debug("Finished Moving data")

