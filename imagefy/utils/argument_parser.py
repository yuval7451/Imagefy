"""
@author: Yuval K aneti
"""
import os
import argparse
from imagefy.utils.common import DIR_DEST, TOP_DEST, TOP_PARAM, VERBOSE_DEST, BASE_PATH_DEST

def arg_parser():
    # create the top-level main_parser
    main_parser = argparse.ArgumentParser(prog='main.py')
    main_parser.add_argument('-d','--dir', action='store', type=str,
                        required=True, help="Path of the image dir", dest=DIR_DEST)

    main_parser.add_argument('-v','--verbose', action='store_true', 
                        required=False, help="Log at INFO or DEBUG", dest=VERBOSE_DEST,
                        default=False)
       
    main_parser.add_argument('-o', '--output', action='store', type=str,
                        help="the output base path for logs, results and more", 
                        dest=BASE_PATH_DEST, default=os.getcwd(), required=True)

    main_parser.add_argument('--top', action='store', type=int, 
                        required=False, help="The number of images that will be taken",
                        dest=TOP_DEST, default=TOP_PARAM)

    args = main_parser.parse_args()
    return args

