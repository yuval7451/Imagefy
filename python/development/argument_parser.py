#! /usr/bin/env python3
import argparse



def arg_parser():
    parser = argparse.ArgumentParser(
        prog="pipe_line.py",
        usage="mode <all\kmeans\inception> dir<path>",
        description="None"
    )
    parser.add_argument(
        "-d",
        "--dir",
        required='true',
        dest="dir"
    )
    parser.add_argument(
        "-m",
        "--mode",
        required='true',
        dest="mode"
    )
    parser.add_argument(
        "-r",
        "--resize",
        default='true',
        dest="resize"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default='true',
        dest="verbose"
    )
    parser.add_argument(
        "-g",
        "-gpu",
        default='true',
        dest="gpu"
    )
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

def validate_parse(args):
    if args.mode == "all" or args.mode == "kmeans":
        assert args.start != None
        assert args.stop != None
