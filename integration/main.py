#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
import asyncio
import numpy as np
from integration.utils.argument_parser import arg_parser
from integration.suits.intergration_suit import IntergrationSuit

#### Functions ####
async def main():
    logging.debug("Starting Main")
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
    await suit.run()
    suit.tensorboard()

if __name__ == '__main__':
    asyncio.run(main())