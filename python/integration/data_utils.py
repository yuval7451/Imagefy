#! /usr/bin/env python3
#Author: Yuval Kaneti⭐

import os
import cv2
import sys
import shutil
import asyncio
import logging
import numpy as np
import concurrent.futures
from abc import ABC
from integration.common import IMAGE_SIZE, IMAGE_TYPES, MAX_WORKERS, OUTPUT_DIR_PATH

class Image():
    """Image -> A Class That holds information about an Image."""
    def __init__(self, src_path : str, data : list):
        """
        @param src_path: C{str} -> The src path of the image
        @param data: C{list} -> A np.array.flatten() containing the image data
        """
        self.src_path = src_path
        self.data = data
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

class DataLoader():
    """DataLoader -> an asyncio Implemntaion of a DataLoader."""
    def __init__(self, dir_path):
        raise NotImplementedError("Module DataLoader is not Implemented yet.\n use DataLoaderWraper() instaed.")
        logging.debug("Initializing DataLoader")
        self.dir_path = dir_path
        self.OutputQueue = asyncio.Queue() 
        self.loop = asyncio.get_event_loop()

    def run(self):
        logging.info("Starting DataLoader")
        ImagesList = []
        self.loop.run_until_complete(self._run())
        self.loop.close()
        while not self.OutputQueue.empty():
            ImagesList.append(self.OutputQueue.get_nowait())
        logging.info(f"Loaded {len(ImagesList)} Images")

        return ImagesList

    async def _run(self):
        image_list = self._list_images()
        InputQueue = asyncio.Queue()
        consumer = asyncio.ensure_future(self._resize_images(InputQueue))
        await self._load_images(InputQueue, image_list)
        await InputQueue.join()
        consumer.cancel()

    async def _load_images(self, InputQueue, image_list):
        for image_path in image_list:
            logging.debug(f"Loading {image_path}")
            # await asyncio.sleep(random.random())
            np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            item = (image_path, np_image)
            await InputQueue.put(item)

    async def _resize_images(self, InputQueue):
        while True:
            image_path, np_image = await InputQueue.get()
            logging.debug(f"resizing {image_path}")
            np_image = np.resize(np_image, (IMAGE_SIZE, IMAGE_SIZE, 3))
            np_image = np_image.flatten()
            await self.OutputQueue.put(Image(src_path=image_path, data=np_image))
            # await asyncio.sleep(random.random())
            InputQueue.task_done()

    def _list_images(self):
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
        logging.info("Starting DataLoader")
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

        logging.info(f"Loaded {len(ImagesList)} Images")
        return ImagesList
        
    def _load_image(self, image_path : str):
        """
        @param image_path: C{str} -> The image path to load.
        @return: C{Image} -> An @Image Object containing information about the image.
        """
        # logging.debug(f"Loading {image_path}")
        np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), dtype=np.uint8)
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
    
