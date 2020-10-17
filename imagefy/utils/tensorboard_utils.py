#! /usr/bin/env python3
"""
Author: Yuval Kaneti
Code Taken from: *http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
                 *https://github.com/nlml/np-to-tf-embeddings-visualiser
"""

#### Imports ####
import os;  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import imageio
import logging 
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

#### Function ####
def save_embeddings(images_features_labels, save_dir):
    """
    @remarks Function to save embeddings (with corresponding labels and images) to a
        specified directory. Point tensorboard to that directory with
        tensorboard --logdir=<save_dir> and your embeddings will be viewable.
    
    @parama images_features_labels: C{dict}
        each key in the dict should be the desired name for that embedding, and 
        each element should be a list of [images, embeddings, labels] where 
        images are a numpy array of images between 0. and 1. of shape [N*W*H*D] 
        or [N*H*W] if grayscale (or None if no images), embeddings is a numpy 
        array of shape [N*D], and labels is a numpy array of something that can
        be converted to string of shape D (or None if no labels available)
    @param save_dir: C{str} -> path to save tensorboard checkpoints
    """
    assert len(list(images_features_labels.keys())), 'Nothing in dictionary!'
    
    # Make directory if necessary
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Reset graph and initialise file writer and session
    tf.compat.v1.reset_default_graph()
    writer = tf.compat.v1.summary.FileWriter(save_dir, graph=None)
    sess = tf.compat.v1.Session()

    config = projector.ProjectorConfig()

    # For each embedding name in the provided dictionary of embeddings
    for name in list(images_features_labels.keys()):
        [images, features, labels] = images_features_labels[name]
    
        # Make a variable with the embeddings we want to visualise
        embedding_var = tf.Variable(features, name=name, trainable=False)    
        # Add this to our config with the image and metadata properties
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # Save sprites and metadata
        if labels is not None:
            projector_filename = f"{name}-metadata.tsv"
            metadata_path = os.path.join(save_dir, projector_filename)
            logging.debug(f"Saving projector metadata to {projector_filename}")
            save_metadata(labels, metadata_path)
            embedding.metadata_path = metadata_path

        if images is not None:
            image_filename = f"{name}.png"
            sprites_path = os.path.join(save_dir, image_filename)
            logging.debug(f"Saving images to {image_filename}")
            save_sprite_image(images, path=sprites_path, invert=len(images.shape)<4)
            embedding.sprite.image_path = sprites_path
            embedding.sprite.single_image_dim.extend(images[0].shape)
    
        # Save the embeddings
        logging.debug(f"Saving Projector Embeddings")
        projector.visualize_embeddings(writer, config)

    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.save(sess, os.path.join(save_dir, 'ckpt'))

def save_labels(labels, name, save_dir):
    """
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logging.info("Saving Labels")
    projector_filename = f"{name}-metadata.tsv"
    metadata_path = os.path.join(save_dir, projector_filename)
    logging.debug(f"Saving projector metadata to {projector_filename}")
    save_metadata(labels, metadata_path)

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. 
       Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))    
    if len(images.shape) > 3:
        spriteimage = np.ones(
            (img_h * n_plots, img_w * n_plots, images.shape[3]))
    else:
        spriteimage = np.ones((img_h * n_plots, img_w * n_plots))
    four_dims = len(spriteimage.shape) == 4
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                if four_dims:
                    spriteimage[i * img_h:(i + 1) * img_h,
                      j * img_w:(j + 1) * img_w, :] = this_img
                else:
                    spriteimage[i * img_h:(i + 1) * img_h,
                      j * img_w:(j + 1) * img_w] = this_img
    return spriteimage
    
def save_sprite_image(to_visualise, path, invert=True):
    """
    """
    if invert:
        to_visualise = invert_grayscale(to_visualise)
    sprite_image = create_sprite_image(to_visualise)

    # sprite_image =  np.array(sprite_image * 255, dtype=np.uint8)
    imageio.imwrite(path, sprite_image)

def invert_grayscale(data):
    """Makes black white, and white black."""
    return 1-data

def save_metadata(batch_ys, metadata_path):
    with open(metadata_path,'w') as f:
        f.write("Index\tLabel\tfilename\n")
        for index, item in enumerate(zip(*batch_ys)):            
            (label, filename) = item
            if type(label) is int:
                f.write("{}\t{}\n".format(index, label))
            else:
                f.write("{}\t{}\t{}\n".format(index, label, filename))
