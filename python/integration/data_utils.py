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
    def __init__(self, src_path, data):
        """
        """
        self.src_path = src_path
        self.basename = os.path.basename(src_path)
        self.dst_dir = None
        self.dst_path = None
        self.cluster_n = None
        self.data = data
        
    def free(self):
        """
        """
        logging.debug(f"Freeing {sys.getsizeof(self.data)} bytes from {self.basename}")
        del self.data

    def flush(self):
        if self.dst_dir:
            self.dst_path = os.path.join(self.dst_dir, self.basename)
        else:
            logging.error("Call Image.flush() First")

class DataLoader():
    def __init__(self, dir_path):
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
    def __init__(self, dir_path, image_size):
        logging.debug("Initializing DataLoader")
        self.dir_path = dir_path
        self.image_size = image_size
    
    def run(self):
        logging.info("Starting DataLoader")
        image_paths = self._list_images()
        ImagesList = []
        #ProcessPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_image = {executor.submit(self.load_image, image_path): image_path for image_path in image_paths}
            for future in concurrent.futures.as_completed(future_to_image):
                try:
                    data = future.result()
                    ImagesList.append(data)
                except Exception as e:
                    logging.critical(e)
                    raise(e)
        
        # ImagesList = np.array(ImagesList)
        logging.info(f"Loaded {len(ImagesList)} Images")
        # logging.info(f"Data shape: {str(ImagesList[0].data.shape)}")
        return ImagesList
        
    def load_image(self, image_path):
        # logging.debug(f"Loading {image_path}")
        np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), dtype=np.uint8)
        # logging.debug(f"resizing {image_path}")
        np_image = np.resize(np_image, (self.image_size, self.image_size, 3))
        np_image = np_image.flatten()
        return Image(src_path=image_path, data=np_image)
        
    def _list_images(self):
        logging.debug("Listing Images")
        image_list = []
        for image_name in os.listdir(self.dir_path):
            if image_name.split('.')[-1].lower() in IMAGE_TYPES:
                image_list.append(os.path.join(self.dir_path, image_name))
        return image_list

class IOWraper():
    """
    """
    def __init__(self, data, score):
        """
        """
        self.data = data
        self.score = score
        self._clean()

    def _clean(self):
        logging.debug("Cleaning Up from last run")
        for _dir in os.listdir(OUTPUT_DIR_PATH):
           shutil.rmtree(os.path.join(OUTPUT_DIR_PATH, _dir))

    def merge_data(self):
        """
        """
        logging.debug("Meging Images & silhouette_score")
        for (image, cluster_label) in zip(self.data, self.score.cluster_labels):
            image.dst_dir = os.path.join(OUTPUT_DIR_PATH, f"cluster_{cluster_label}")
            image.cluster_n = int(cluster_label)
            # image.free()
            image.flush()

    def create_output_dirs(self):
        """
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
                    # raise(e)
        logging.debug("Finished Creating Output Directories")

    def move_data(self):
        """
        """
        logging.debug("Moving Data")
        #ThreadPoolExecutor
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
    """
    """
    def __init__(self, n_clusters, cluster_labels):
        """
        """
        self.n_clusters = n_clusters
        self.cluster_labels = cluster_labels
    
