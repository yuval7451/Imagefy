#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
import time, datetime
from imagefy.utils.argument_parser import arg_parser
from imagefy.suits.integration_suit import IntegrationSuit

#### Functions ####
def main():
    start = time.time()
    t = datetime.datetime.now() 
    logging.info(f"Starting Main at {t}")
    args = arg_parser()
    suit = IntegrationSuit(**vars(args))
    suit.run()
    end  = time.time() - start
    t = datetime.datetime.now()
    logging.info(f"Finished Running Main at {t}")
    logging.info(f"Program took {end} Seconds to run")

if __name__ == '__main__':
    main()