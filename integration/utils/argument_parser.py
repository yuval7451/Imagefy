#! /usr/bin/env python3

import argparse
from integration.utils.common import IMAGE_SIZE
def arg_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required='true',
        dest="dir"
    )
    parser.add_argument(
        "-s",
        "--size",
        default=IMAGE_SIZE,
        dest="size"
    )
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     default='true',
    #     dest="verbose"
    # )
    parser.add_argument(
        "-start",
        dest="start"
    )
    parser.add_argument(
        "-stop",
        dest="stop"
    )

    args = parser.parse_args()
    return args
