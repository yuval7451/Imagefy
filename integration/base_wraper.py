#! /usr/bin/env python3
"""
Author: Yuval Kaneti⭐
"""
#### IMPORTS ####
from abc import ABC, abstractmethod

class BaseWraper(ABC):
    """BaseWraper -> An Abstract Class for TensorflowWrapers."""
    def __init__(self):
        pass
    
    @abstractmethod
    def run(self):
        pass