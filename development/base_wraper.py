#! /usr/bin/env python3
"""
Author: Yuval Kaneti⭐
"""
#### IMPORTS ####
from abc import ABC, abstractmethod

class BaseWraper(ABC):
    """
    """
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def start(self):
        pass
      
    def __str__(self):
        pass