#!/usr/bin/env python3
# Author: Yuval Kaneti

## Imports
import json
import shutil
import logging
import numpy as np
from glob import glob
from time import time
from tqdm import tqdm
import tensorflow as tf
from typing import List
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from imagefy.utils.argument_parser import arg_parser
from imagefy.utils.data_utils import input_fn, Image, move_images
from imagefy.utils.common import IFERENCE_MODEL_DIR, INCEPTION_RESNET_INFERENCE_DENSE, INCEPTION_RESNET_INFERENCE_INPUT, SERVING_DEFAULT

## Logging Related
StreamHandler = logging.StreamHandler()
StreamHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(name)s - %(filename)s - %(funcName)s - %(asctime)s - %(message)s')
StreamHandler.setFormatter(formatter)

logger = logging.getLogger("Imagefy")
logger.addHandler(StreamHandler)
logger.setLevel(logging.DEBUG)

logging.getLogger('tensorflow').addHandler(StreamHandler)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


## Functions 
def main():
    start = time()
    args = arg_parser()
    data_dir = args.dir_path
    dest_dir = args.base_path
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # else:
        ## NOTE Be Carfull With Removing the Directory
        # shutil.rmtree(dest_dir)
        # os.makedirs(dest_dir)

    logger.info("Starting Main")
    images = predict_all(data_dir_regex=data_dir, dest_dir=dest_dir)
    save_predictions(images, dest_dir)
    top_images = images[:args.top]
    logger.info("Start to move Images")
    move_images(images=top_images)
    logger.info("Finished Main")
    end = time() - start
    logger.info(f"it took {end} Seconds to run.")


def predict_all(data_dir_regex: str, dest_dir: str) -> List:
    """
    @param: data_dir_regex: C{str} -> The data dir regex 
    @param: dest_dir: C{str} -> The dest dir path
    """
    logger.debug("Loading inference model")
    predictor = tf.saved_model.load(IFERENCE_MODEL_DIR) 
    images = []
    data_dir = data_dir_regex.split("*")[-2]
    logger.info("Starting to make predictions")
    for T_image, T_image_name in tqdm(input_fn(data_dir_regex=data_dir_regex), total=len(glob(data_dir_regex))):
        image = T_image.numpy().flatten().tolist()
        image_name = T_image_name.numpy().decode()
        example = tf.train.Example()
        example.features.feature[INCEPTION_RESNET_INFERENCE_INPUT].float_list.value.extend(image) # type: ignore
        output_dict = predictor.signatures[SERVING_DEFAULT](predictor_inputs=tf.constant([example.SerializeToString()])) # type: ignore
        y_predicted = output_dict[INCEPTION_RESNET_INFERENCE_DENSE][0].numpy()
        si = np.arange(1, 11, 1)
        score = np.sum(y_predicted * si)
        images.append(Image(image_name=image_name, score=score, data_dir=data_dir, dest_dir=dest_dir))
    
    logger.info("Finished makeing predictions")
    return sorted(images, key=lambda image: image.score, reverse=True)


def save_predictions(images: List[Image], dest_dir:str):
    logger.info("Saving predictions to file")
    predictions = {}
    for image in images:
        predictions[image.image_name] = {"score": image.score, "src_path": image.src_path, "dest_path": image.dest_path}

    results_path = os.path.join(dest_dir, "results.json")
    with open(results_path, "w") as fd:
        fd.write(json.dumps(predictions))


if __name__ == '__main__':
    main()