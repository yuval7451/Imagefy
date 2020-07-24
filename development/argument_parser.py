#! /usr/bin/env python3
import argparse



def arg_parser():
    parser = argparse.ArgumentParser(
        prog="pipe_line.py",
        usage="<mode>",
        description="None"
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        dest="dir"
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        dest="mode"
    )
    parser.add_argument(
        "-r",
        "--resize",
        default=True,
        dest="resize"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        dest="verbose"
    )
    parser.add_argument(
        "-g",
        "-gpu",
        default=True,
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
    if args.mode == "all" or args.mode == "cluster":
        assert args.start != None
        assert args.stop != None
