#!/usr/bin/env python3
# Author: Yuval Kaneti⭐

#### Imports ####
import logging
import numpy as np
from integration.utils.argument_parser import arg_parser
from integration.suits.intergration_suit import IntergrationSuit
from integration.utils.enum import Enum
#### Functions ####

def main():
    logging.info("Starting Main")
    args = arg_parser()
    suit = IntergrationSuit(**vars(args))
    suit.run()

if __name__ == '__main__':
    main()