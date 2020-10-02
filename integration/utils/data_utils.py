#! /usr/bin/env python3
#Author: Yuval Kaneti⭐

import os
import cv2
import sys
import time
import shutil
import asyncio
import logging
import numpy as np
import concurrent.futures
from abc import ABC
from integration.utils.common import IMAGE_TYPES, MAX_WORKERS, OUTPUT_DIR_PATH, TENSORBOARD_LOG_DIR
from integration.utils.tensorboard_utils import save_embeddings

class Image():
    """Image -> A Class That holds information about an Image."""
    def __init__(self, src_path : str, data : list):
        """
        @param src_path: C{str} -> The src path of the image
        @param data: C{list} -> A np.array.flatten() containing the image data
        """
        self.src_path = src_path
        self.data = data
        self.shape = self.data.shape
        self.basename = os.path.basename(src_path)
        self.dst_dir = None
        self.dst_path = None
        self.cluster_n = None
        
    def free(self):
        """
        @remarks *Deletes the image data from memory.
                 *Should be Tested to make sure it actually does somthing.
        """
        logging.debug(f"Freeing {sys.getsizeof(self.data)} bytes from {self.basename}")
        del self.data

    def flush(self):
        """
        @remarks *Only if self.dst_dir was chosen the dst_path can be filled.
                 *Can only be used after IOWraper.merge_data() was invoked.
        """
        if self.dst_dir:
            self.dst_path = os.path.join(self.dst_dir, self.basename)
        else:
            logging.error("Call IOWraper.merge_data() First")

class BaseScore(ABC):
    """BaseScore -> An Abstarct Class represnting a BaseWraper Return Value."""
    def __init__(self, n_clusters : int, cluster_labels: list):
        """
        @param n_cluster: C{int} -> The nmber of clusters.
        @param cluster_labels: C{list} -> the corosponding cluster index ordered by the data originial order.
        @remarks *DO NOT USE THIS CLASS, inherite From it.
        """
        self.n_clusters = n_clusters
        self.cluster_labels = cluster_labels
    
class DataLoader():
    """DataLoader -> an asyncio Implemntaion of a DataLoader."""
    def __init__(self, dir_path, image_size):
        # raise NotImplementedError("Module DataLoader is not Implemented yet.\n use DataLoaderWraper() instaed.")
        """
        @dir_path: C{str} -> The dir to load the pictures from.
        @image_size: C{int} -> The size the images should be resized to befor flattening them [size, size, 3] -> [size * size * 3]
        """
        logging.debug("Initializing DataLoader")
        self.dir_path = dir_path
        self.image_size = image_size
        self._loop = asyncio.get_running_loop()
        self._semaphore = asyncio.Semaphore(MAX_WORKERS)
    
    async def run(self):
        """
        """
        logging.info("Running AsyncIO DataLoader")
        start = time.time()
        # # asyncio.run(self._run())
        # await self._loop.run_until_complete(self._run())
        # self._loop.close()
        # return ImagesList
        image_paths = self._list_images()
        ImagesList =  await asyncio.gather(
            *[self._load_image(image_path) for image_path in image_paths]
        )
        end =  time.time() - start
        logging.info(f"It took {end} Seconds To Load {len(ImagesList)}")
        return ImagesList

    async def _load_image(self, image_path : str):
        """
        @param image_path: C{str} -> The image path to load.
        @return: C{Image} -> An @Image Object containing information about the image.
        """        
        # async with self._semaphore:
        logging.debug(f"Loading {image_path}")
        np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), dtype=np.uint8)
        logging.debug(f"resizing {image_path}")
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

class DataLoaderWraper():
    """DataLoaderWraper -> a Threaded IO & Numpy Opertions Wraper."""
    def __init__(self, dir_path : str, image_size: int):
        """
        @dir_path: C{str} -> The dir to load the pictures from.
        @image_size: C{int} -> The size the images should be resized to befor flattening them [size, size, 3] -> [size * size * 3]
        """
        logging.debug("Initializing DataLoader")
        self.dir_path = dir_path
        self.image_size = image_size
    
    def run(self):
        """
        @remarks *The only function that the user should call.
                 *Will start the invoketion of ThreadPoolExecutor(@MAX_WORKERS).
                 *There is an option to use ProcessPoolExecutor but the number of @MAX_WORKERS will have to be lower. 
                 *If an error acourd it will be loged & raised.
        """
        logging.info("Starting DataLoaderWraper")
        start = time.time()
        image_paths = self._list_images()
        ImagesList = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_image = {executor.submit(self._load_image, image_path): image_path for image_path in image_paths}
            for future in concurrent.futures.as_completed(future_to_image):
                try:
                    data = future.result()
                    ImagesList.append(data)
                except Exception as e:
                    logging.critical(e)
                    raise(e)

        # logging.info(f"Loaded {len(ImagesList)} Images")
        end = time.time() - start
        logging.info(f"It took {end} Seconds To Load {len(ImagesList)} Images")
        return ImagesList
        
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

class IOWraper():
    """IOWraper -> An Object which handles multithreaded IO Opartions."""
    def __init__(self, data : list, score : BaseScore):
        """
        @param data: C{np.ndarray} -> The Loaded data.
        @param score: C{BaseScore} -> The Output returned by on of the @BaseWrapers.
        @remarks *@self.score can be any of the inherited object from BaseScore.
                 *@self._clean is invoked on __init__, it will remove all the old cluster output directories.    
        """
        self.data = data
        self.score = score
        self._clean()

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
        @remarks *will use the data from self.score (@BaseScore object) to set the Image.dst_fir & Image.cluster_n.
                 *This function must be called befor @IOWraper.move_data().
                 *A usual use case will look like this:
                  IOWraper.create_output_dirs()
                  IOWraper.merge_data()
                  IOWraper.move_data()
        """
        logging.debug("Meging Images & BaseScore")
        for (image, cluster_label) in zip(self.data, self.score.cluster_labels):
            image.dst_dir = os.path.join(OUTPUT_DIR_PATH, f"cluster_{cluster_label}")
            image.cluster_n = int(cluster_label)
            # image.free()
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
            for dir_index in range(self.score.n_clusters):
                dir_path = os.path.join(OUTPUT_DIR_PATH, f"cluster_{dir_index}")
                future_to_dir[executor.submit(os.mkdir, dir_path)] = dir_index

            for future in concurrent.futures.as_completed(future_to_dir):
                try:
                    result = future.result()
                except Exception as e:
                    logging.error(e)
                    raise(e)
        logging.debug("Finished Creating Output Directories")

    def move_data(self):
        """
        @remarks *Will Copy The Images from Image.src_path to Image.dst_path Using shutil & ThreadPoolExecutor(@MAX_WORKERS)
                 *@IOWraper.merge_data MUST be invoked befor this function or a RuntimeError will be raised.
                 *Once shutil.copy is invoked with image.dst_path = None. No Warning is given at this stage.
        """
        logging.debug("Moving Data")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file_moved = {executor.submit(shutil.copy, image.src_path, image.dst_path): image for image in self.data}
            for future in concurrent.futures.as_completed(future_to_file_moved):
                try:
                    result = future.result()
                except Exception as e:
                    logging.error(e)
                    raise(e)
        logging.debug("Finished mooving data")

class TensorboardWraper():
    """TensorboardWraper -> A Class that will generate Tensorboard projector Files."""
    def __init__(self, data : list, X: np.ndarray=None, y : list=None):
        """
        @param data: C{list} -> A List of Image Objects.
        @param X: Optional C{np.ndarray} -> A np.ndarray of Image.data Arrrays.
        @param y: Optional C{list} -> A list of Image.n_cluster, Could be improved into cluster_{cluster_n}
        @remarks: *Normal usage will provide the data varibale and X,y will be extracted via self._image_to_nd,
                       In the Future Image object itself will have this builtin.
                  *When Using TensorboardWraper.load(), on initiliazition, specify data=None, a warning will be logged,
                       It will remind you to use TensorboardWraper.load() because no data is loaded.

        """
        self.data = data
        if X is not None and y is not None:
            self.X = X
            self.y = y
        elif self.data is not None:
            self._image_to_nd()
        else:
            logging.warn("Please Make sure to use TensorboardWraper.load(), Data is currently None")

    def save(self):
        """
        @remarks: *Save self.data & self.labels to .npy Files for future use.
                    *Can be loaded via @IntergrationSuit.visualize_from_file() OR @self.load() -> Should not be USED!
        """
        if self.y is not None and self.X is not None:        
            filenameX = f"X_{self.data[0].shape}.npy"
            filenameY = f"Y_{self.data[0].shape}.npy"
            logging.info(f"Saving Clustering X Data to: {filenameX}")
            np.save(filenameX, self.X)
            logging.info(f"Saving Clustering Data to: {filenameY}")
            np.save(filenameY, self.y)

    def load(self, filename : str):
        """
        @param filename: The base Filename (aka (60,784).npy)
        @remarks *it will look for X_filename & Y_filename, (DON'T Specify 'X_' or 'Y_')!
        """
        logging.info(f"Loading X_{filename} & Y_{filename}")
        self.X = np.load(f"X_{filename}")
        self.y = np.load(f"Y_{filename}")
        logging.info("You can call Visualize3D.show()")

    def _image_to_nd(self):
        """
        @param data: C{np.ndarray} -> an array on Image Objects.
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

    def create_tensorboard_output(self, name : str="Imagefy"):
        """
        @param name: C{str} -> The Tensor name, doesnt really have a meaning in this context.
        @remarks *Might launch a subprocess that will run Tensorboard & open Chrome in the futre
        """
        logging.info("Creating Tensorboard metadata")
        images_features_labels = {}
        images_features_labels[name] = [None, self.X, self.y]
        logging.info("Saving Embeddings")
        log_name = f"{name}-{time.time()}"
        save_embeddings(images_features_labels, os.path.join(os.getcwd(),TENSORBOARD_LOG_DIR, log_name))
        logging.info(f"Run tensorboard --logdir={TENSORBOARD_LOG_DIR + log_name}")
        logging.info("Go to http://localhost:6006/ and click Projector")
