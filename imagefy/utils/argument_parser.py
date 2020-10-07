#! /usr/bin/env python3
import os
import argparse
from imagefy.utils.common import IMAGE_SIZE, MINI_KMEANS_NUM_EPOCHS, EPOCHS_DEST, NUM_ITERATION_DEST, \
    SAVE_MODEL_DEST, DATA_LOADER_DEST, LOADER_DEST, LOADER_OPTIONS, DIR_DEST, VERBOSE_DEST, \
    SIZE_DEST, TENSORBOARD_DEST, KMEANS_DEST, START_DEST, END_DEST, MINI_KMEAND_DEST, NUM_CLUSTERS_DEST, \
    BATCH_SIZE_DEST, MINI_KMEANS_BATCH_SIZE, BASE_PATH_DEST

def arg_parser():
    # create the top-level main_parser
    main_parser = argparse.ArgumentParser(prog='main.py')
    main_parser.add_argument('-d','--dir', action='store', type=str,
                        required=True, help="Path of the image dir", dest=DIR_DEST)

    main_parser.add_argument('-v','--verbose', action='store_true', 
                        required=False, help="Log at INFO or DEBUG", dest=VERBOSE_DEST,
                        default=False)

    main_parser.add_argument('-s','--size', action='store', type=int, 
                        required=False, help="Resize Images to <SIZE>", dest=SIZE_DEST,
                        default=IMAGE_SIZE)

    main_parser.add_argument('-t','--tensorboard', action='store_true',
                        help="Save Tensorboard output", dest=TENSORBOARD_DEST)
       
    main_parser.add_argument('-l', '--loader', action='store', type=str, 
                        required=True, help="Should we Save he trained model",
                        default=DATA_LOADER_DEST, dest=LOADER_DEST, choices=LOADER_OPTIONS)

    main_parser.add_argument('-o', '--output', action='store', type=str,
                        help="the output base path for logs, results and more", 
                        dest=BASE_PATH_DEST, default=os.getcwd(), required=True)
    
    
    sub_parsers = main_parser.add_subparsers(help='Which Wraper to use') 

    #Sub Parsers
    kmeans_sub_parser(sub_parsers=sub_parsers)
    mini_batch_kmeans_sub_parser(sub_parsers=sub_parsers)
    args = main_parser.parse_args()
    return args

def kmeans_sub_parser(sub_parsers): #parents=[main_parser]
    kmeans_parser = sub_parsers.add_parser(KMEANS_DEST, help='Arguments for the Kmeans Wraper', add_help=False)

    kmeans_parser.add_argument('--start', action='store', type=int, 
                        required=True, help="The startig number of neighbors", dest=START_DEST)

    kmeans_parser.add_argument('--end', action='store', type=int, 
                        required=True, help="The end number of neighbors", dest=END_DEST)

    kmeans_parser.add_argument('-i','--iter', action='store', type=int, 
                        required=True, help="The number of iterations of training <1000 ish>", dest=NUM_ITERATION_DEST)
                        
    kmeans_parser.set_defaults(wraper=KMEANS_DEST)

def mini_batch_kmeans_sub_parser(sub_parsers):
    mini_kmeans_parser = sub_parsers.add_parser(MINI_KMEAND_DEST, help='Arguments for the Mini Batch Kmeans Wraper', add_help=False)

    mini_kmeans_parser.add_argument('-e', '--epochs', action='store', type=int, 
                        required=True, help="The number of epochs", dest=EPOCHS_DEST,
                        default=MINI_KMEANS_NUM_EPOCHS)
    
    mini_kmeans_parser.add_argument('-b', '--batch_size', action='store', type=int, 
                        required=True, help="The batch size", dest=BATCH_SIZE_DEST,
                        default=MINI_KMEANS_BATCH_SIZE)

    mini_kmeans_parser.add_argument('-c','--num_clusters', action='store', type=int, 
                        required=True, help="The number of clusters in Mini Batch Kmeans", dest=NUM_CLUSTERS_DEST)

    mini_kmeans_parser.add_argument('--save', action='store', type=bool, 
                        required=False, help="Should we Save he trained model",
                        default=False, dest=SAVE_MODEL_DEST)

    mini_kmeans_parser.set_defaults(wraper=MINI_KMEAND_DEST)


