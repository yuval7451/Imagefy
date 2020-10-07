#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import time, datetime
import logging
from imagefy.utils.argument_parser import arg_parser
from imagefy.suits.integration_suit import IntegrationSuit
#### Functions ####

def main():
    start = time.time()
    args = arg_parser()
    suit = IntegrationSuit(**vars(args))
    suit.run()
    end  = time.time() - start
    logging.info(f"Program took {end} Seconds to run")

if __name__ == '__main__':
    main()