#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import time, datetime
import logging
import numpy as np
from imagefy_clustering.utils.argument_parser import arg_parser
from imagefy_clustering.suits.intergration_suit import IntergrationSuit
# from imagefy_clustering.utils.enum import Enum
#### Functions ####

def main():
    start = time.time()
    t = datetime.datetime.now() #.strftime("%Y%m%d-%H%M%S")
    logging.info(f"Starting Main at {t}")
    args = arg_parser()
    suit = IntergrationSuit(**vars(args))
    suit.run()
    end  = time.time() - start
    t = datetime.datetime.now() #.strftime("%Y%m%d-%H%M%S")
    logging.info(f"Finished Running Main at {t}")
    logging.info(f"Program took {end} Seconds to run")

if __name__ == '__main__':
    main()