#! /usr/bin/env python3
"""
Author: Yuval Kaneti⭐
Code Taken From: https://github.com/titu1994/neural-image-assessment/blob/master/
"""

#### IMPROTS ####
import os
import cv2
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from utils.score_utils import mean_score, std_score
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


# Keras Model To estimator
def load_modal(device, weights_path):
    """
    :return model: C{tensorflow.keras.models.Model} -> The Loaded Model
    """   
    logging.debug(f"Using {device}")
    logging.info("Loading Model")
    with tf.device(device):
        base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
        # Old Model
        # x = Dropout(0.75)(base_model.output)
        # x = Dense(10, activation='softmax')(x)
        x = Dense(10, activation='softmax')(base_model.output)
        model = Model(base_model.input, x)
        model.load_weights(weights_path)
        model.compile(optimizer="Adam", loss="mse", metrics=["accuracy", "mae"])
    
    
    return model

def model_to_estimator(model):
    logging.info("Converting model to estimator")
    gpu_options = tf.compat.v1.GPUOptions(
        allow_growth=True,
        force_gpu_compatible=True,
        per_process_gpu_memory_fraction=0.9,
    )
        
    config_proto = tf.compat.v1.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False,
    )

    config = tf.estimator.RunConfig(
        model_dir="D:\\Imagefy\\resources\\models\\InceptionResNetV2",
        save_summary_steps=5,
        keep_checkpoint_max=1,
        session_config=config_proto
    )

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, keras_model_path=None, custom_objects=None,
        config=config, checkpoint_format='checkpoint'
    )

    return estimator

# Estimator to infrence
def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[1], name='input_tensors')
    receiver_tensors      = {"predictor_inputs": serialized_tf_example}
    feature_spec          = {"input_1": tf.FixedLenFeature(shape=[224, 224, 3], dtype=tf.float32)}
    features              = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def save(estimator, input_fn):
    model_path = os.path.join("D:\\Imagefy\\resources\\models\\InceptionResNetV2", 'inference')
    logging.info(f"Saving model to {model_path}")
    estimator.export_saved_model(model_path, input_fn)

# Infrence
def load_images(folder_path):
    images = []
    size = 224
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        np_image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), dtype=np.float32)
        np_image = np.resize(np_image, (size, size, 3))
        # Scale -1 & 1
        np_image = np_image - np_image.mean(axis=0)
        np_image = np_image / np.abs(np_image).max(axis=0)
        # np_image = np.expand_dims(np_image, axis=0)
        images.append(np_image.flatten().tolist())
    images = np.asarray(images)
    return images

def inference():
    with tf.device('/device:GPU:0'):    
        images = load_images("D:\\Imagefy\\dataset\\inference")
        full_model_dir = "D:\\Imagefy\\resources\\models\\InceptionResNetV2\\inference\\1601923417"
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
            predictor   = tf.contrib.predictor.from_saved_model(full_model_dir)
            for image in images: 
                model_input = tf.train.Example(features=tf.train.Features( feature={"input_1": tf.train.Feature(float_list=tf.train.FloatList(value=image)) })) 
                model_input = model_input.SerializeToString()
                output_dict = predictor({"predictor_inputs":[model_input]})
                y_predicted = output_dict["dense"][0]
                mean = mean_score(y_predicted)
                std = std_score(y_predicted)
                print(f"NIMA Score : {mean}, {std}")
                    
def inference_dataset(dataset):
    results = {} 
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_element = iterator.get_next()
    full_model_dir = "D:\\Imagefy\\resources\\models\\InceptionResNetV2\\inference\\1601923417"
    gpu_options = tf.compat.v1.GPUOptions(
        allow_growth=True,
        force_gpu_compatible=True,
        per_process_gpu_memory_fraction=0.9,
    )
    config_proto = tf.compat.v1.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False,
    )
    
    with tf.compat.v1.Session(config=config_proto) as sess:
        try:
        # imported = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
            predictor = tf.contrib.predictor.from_saved_model(full_model_dir)
            while True:
                with tf.device('/device:GPU:0'):    
                    image, label, image_name = sess.run(next_element)
                    label = label.decode()  
                    image_name = image_name.decode() 
                    print(label, image_name)
                    image = image.flatten().tolist()
                    model_input = tf.train.Example(features=tf.train.Features( feature={"input_1": tf.train.Feature(float_list=tf.train.FloatList(value=image)) })) 
                    model_input = model_input.SerializeToString()
                    output_dict = predictor({"predictor_inputs":[model_input]})
                    y_predicted = output_dict["dense"][0]
                mean = mean_score(y_predicted)
                std = std_score(y_predicted)
                print(f"NIMA Score : {mean}, {std}")
                if label in results.keys():
                    results[label].append({image_name: mean})
                else:
                    results[label] = [({image_name: mean})]
        
        except tf.errors.OutOfRangeError:
            logging.info("Finished Loading Data")

    return results


# Dataset API Didnt Use
@tf.function
def inception_input_fn(output_dir_path):    
    options = tf.data.Options()
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.autotune_cpu_budget = True
    dataset = tf.data.Dataset.list_files(os.path.join(output_dir_path, "*", "*"), shuffle=False)
    dataset = dataset.map(_load_tensor_with_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)#.cache()   
    return dataset 
    
@tf.function
def _load_tensor_with_label(image_path: str):
    """
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
    

def main():
    # PATH = os.path.join(DATA_FOLDER_PATH, GOPRO_IMAGES_FOLDER)
    # weights_path = os.path.join(WEIGHTS_FOLDER_PATH, INCEPTION_RESNET_WEIGHTS)
    # ctarget_size = (IMAGE_SIZE, IMAGE_SIZE)
    # device = '/GPU:0'  
    # model = _load_modal(device='/GPU:0', weights_path=weights_path)
    # estimator = model_to_estimator(model)
    # save(estimator=estimator, input_fn=serving_input_receiver_fn)
    # Normal infrence
    # inference()
    # Dataset API Infrence
    dataset = inception_input_fn("D:\\Imagefy\\results\\output\\KmeansTensorflowWraper-2020-10-06--15-12-47")
    results = inference_dataset(dataset)
    # get_top_k(results, k=3)
    print(results)
if __name__ == '__main__':
    main()