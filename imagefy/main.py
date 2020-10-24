# Author: Yuval Kaneti

## Imports 
import logging 
import time
from imagefy.utils.argument_parser import arg_parser
from imagefy.suits.intergration_suit import IntergrationSuit

## Functions 
def main():
    start = time.time()

    args = arg_parser()
    suit = IntergrationSuit(**vars(args))
    suit.run()

    end  = time.time() - start
    logging.info(f"Program took {end} Seconds to run")

if __name__ == '__main__':
    main()