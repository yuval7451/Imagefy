#! /usr/bin/env python3

import argparse
from integration.utils.common import IMAGE_SIZE, MINI_KMEANS_NUM_ITERATIONS, ITER_DEST, TENSORBOARD_NAME_PARAM, TENSORBOARD_NAME_DEST, SAVE_MODEL_DEST
from integration.utils.common import DIR_DEST, VERBOSE_DEST, SIZE_DEST, TENSORBOARD_DEST, KMEANS_DEST, START_DEST, END_DEST, MINI_KMEAND_DEST, ITER_DEST, NUM_CLUSTERS_DEST
#     args = main_parser.parse_args()
#     return args

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
    
    main_parser.add_argument('-l', '--tensor_log', action='store',
                        help="Save Tensorboard output", dest=TENSORBOARD_NAME_DEST, default=TENSORBOARD_NAME_PARAM)
    
    #
    sub_parsers = main_parser.add_subparsers(help='Which Wraper to use') #dest='wraper'

    #Sub Parsers
    kmeans_sub_parser(sub_parsers=sub_parsers)
    mini_batch_kmeans(sub_parsers=sub_parsers)
    #aliases=['co']
    args = main_parser.parse_args()
    return args

def kmeans_sub_parser(sub_parsers): #parents=[main_parser]
    kmeans_parser = sub_parsers.add_parser(KMEANS_DEST, help='Arguments for the Kmeans Wraper', add_help=False)

    kmeans_parser.add_argument('--start', action='store', type=int, 
                        required=True, help="The startig number of neighbors", dest=START_DEST)

    kmeans_parser.add_argument('--end', action='store', type=int, 
                        required=True, help="The end number of neighbors", dest=END_DEST)
    kmeans_parser.set_defaults(wraper='kmeans')

def mini_batch_kmeans(sub_parsers):
    mini_kmeans_parser = sub_parsers.add_parser(MINI_KMEAND_DEST, help='Arguments for the Mini Batch Kmeans Wraper', add_help=False)

    mini_kmeans_parser.add_argument('-i', '--iter', action='store', type=int, 
                        required=True, help="The number of iteration", dest=ITER_DEST,
                        default=MINI_KMEANS_NUM_ITERATIONS)

    mini_kmeans_parser.add_argument('-c','--num_clusters', action='store', type=int, 
                        required=True, help="The number of clusters in Mini Batch Kmeans", dest=NUM_CLUSTERS_DEST)

    mini_kmeans_parser.add_argument('--save', action='store', type=bool, 
                        required=False, help="Should we Save he trained model",
                        default=False, dest=SAVE_MODEL_DEST)


    # SAVE_MODEL_DEST
    mini_kmeans_parser.set_defaults(wraper="mini-kmeans")

"""
name or flags - Either a name or a list of option strings, e.g. foo or -f, --foo.
action - The basic type of action to be taken when this argument is encountered at the command line.
nargs - The number of command-line arguments that should be consumed.
const - A constant value required by some action and nargs selections.
default - The value produced if the argument is absent from the command line.
type - The type to which the command-line argument should be converted.
choices - A container of the allowable values for the argument.
required - Whether or not the command-line option may be omitted (optionals only).
help - A brief description of what the argument does.
metavar - A name for the argument in usage messages.
dest - The name of the attribute to be added to the object returned by parse_args().
"""

"""
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--parent', type=int)

foo_parser = argparse.ArgumentParser(parents=[parent_parser])
foo_parser.add_argument('foo')
foo_parser.parse_args(['--parent', '2', 'XXX'])
Namespace(foo='XXX', parent=2)

bar_parser = argparse.ArgumentParser(parents=[parent_parser])
bar_parser.add_argument('--bar')
bar_parser.parse_args(['--bar', 'YYY'])
Namespace(bar='YYY', parent=None)

"""