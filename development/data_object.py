#!/usr/bin/env python3
# Author: Yuval Kaneti

class DataObject(object):
    """
    """
    def __init__(self, path, image):
        """
        """
        self.path = path
        self.image = image
        self.image_flatten = self.image.flatten()