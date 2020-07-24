#https://github.com/titu1994/neural-image-assessment/blob/master/evaluate_inception_resnet.py
import os
import numpy as np
import argparse
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from utils.score_utils import mean_score, std_score

from utils.common import WEIGHTS_FOLDER_PATH, INCEPTION_RESNET_WEIGHTS

# 'weights/inception_resnet_weights.h5'
WEIGHTS_PATH = os.path.join(WEIGHTS_FOLDER_PATH, INCEPTION_RESNET_WEIGHTS)

parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it')

parser.add_argument('-img', type=str, default=[None], nargs='+',
                    help='Pass one or more image paths to evaluate them')

parser.add_argument('-resize', type=str, default='false',
                    help='Resize images to 224x224 before scoring')

parser.add_argument('-rank', type=str, default='true',
                    help='Whether to tank the images after they have been scored')

args = parser.parse_args()
resize_image = args.resize.lower() in ("true", "yes", "t", "1")
target_size = (224, 224) if resize_image else None
rank_images = args.rank.lower() in ("true", "yes", "t", "1")

# give priority to directory
if args.dir is not None:
    print("Loading images from directory : ", args.dir)
    imgs = [os.path.join(args.dir, img_name) for img_name in os.listdir(args.dir)]
elif args.img[0] is not None:
    print("Loading images from path(s) : ", args.img)
    imgs = args.img

else:
    raise RuntimeError('Either -dir or -img arguments must be passed as argument')

# with tf.device('/CPU:0'):

with tf.device('/GPU:0'):
    base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights(WEIGHTS_PATH)

    score_list = []

    for img_path in imgs:
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        scores = model.predict(x, batch_size=1, verbose=0)[0]
        mean = mean_score(scores)
        std = std_score(scores)
        score_list.append((img_path, mean))

        print("Evaluating : ", img_path)
        print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
        print()

    if rank_images:
        print("*" * 40, "Ranking Images", "*" * 40)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(score_list):
            print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))

