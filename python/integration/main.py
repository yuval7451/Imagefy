#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
import asyncio
from integration.argument_parser import arg_parser
from integration.intergration_suit import IntergrationSuit
from integration.testing_suit import TesingSuit

#### Functions ####
def main():
    args = arg_parser()
    dir_path = str(args.dir)
    image_size = int(args.size)
    verbose = False
    start_k = int(args.start) if args.start else None
    stop_k = int(args.stop) if args.stop else None

    # Debug
    if verbose: logging.getLogger().setLevel(logging.DEBUG) 
    else: logging.getLogger().setLevel(logging.INFO)
   
    suit = IntergrationSuit(dir_path=dir_path, start_k=start_k, stop_k=stop_k, image_size=image_size)
    suit.run()
    suit.visualize()
    # suit.visualize_from_file("PCA_80_3D.npy")

    # logging.info("Running Testing Suite")
    # suit = TesingSuit(dir_path=dir_path, start_k=start_k, stop_k=stop_k, image_size=image_size)
    # suit.run()
    # suit.visualize()

if __name__ == '__main__':
    main()