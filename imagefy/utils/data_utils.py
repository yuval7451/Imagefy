#! /usr/bin/env python3
#Author: Yuval Kaneti

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
from tqdm import tqdm
import tensorflow as tf
import concurrent.futures
from imagefy.utils.common import MAX_WORKERS, INCEPTION_RESNET_IMAGE_SIZE

class Image():
    def __init__(self, image_name: str, score: float, data_dir: str, dest_dir: str):
        self.image_name = image_name
        self.score = score
        self.src_path = os.path.join(data_dir, self.image_name)
        self.dest_path = os.path.join(dest_dir, self.image_name)

    def __str__(self):
        return f"{self.image_name}: -> {self.score}"
    
    def __repr__(self):
        return f"{self.image_name}: -> {self.score}"

# Data Loading
def input_fn(data_dir_regex):
    """
    @return tf.data.Dataset -> the dataset for Inception
    """
    dataset = tf.data.Dataset.list_files(data_dir_regex, shuffle=False)
    options = tf.data.Options()
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.autotune_cpu_budget = True
    dataset = dataset.map(_load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.copy_to_device("/device:GPU:0"))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)   
    dataset = dataset.with_options(options)
    return dataset

@tf.function
def _load_tensor(image_path: str):
    """
    @param image_path: C{tf.Tensor} -> The image path Tensor, created by tf.data.Dataset.list_files(...).
    @return C{tf.Tensor} -> A tensor image, normalized (-1, 1) & flattended.
    """   
    image = tf.io.read_file(image_path)
    image_name = tf.strings.split(image_path, os.sep)[-1]
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[INCEPTION_RESNET_IMAGE_SIZE, INCEPTION_RESNET_IMAGE_SIZE])
    image = image - tf.math.reduce_mean(input_tensor=image, axis=0)
    image = image / tf.math.reduce_max(input_tensor=tf.abs(image), axis=0)
    image = tf.reshape(image, [-1])
    return image, image_name

def move_images(images):   
    with tqdm(total=len(images)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file_moved = [executor.submit(shutil.copy, image.src_path, image.dest_path) for image in images]
            for future in concurrent.futures.as_completed(future_to_file_moved):
                    _ = future.result()
                    pbar.update(1)