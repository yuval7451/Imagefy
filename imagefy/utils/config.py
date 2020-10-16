"""
@Author: Yuval Kaneti 
"""

from imagefy.utils.common import BASE_PATH_PARAM, INFERENCE_MODEL_DIR_PARAM, SIZE_PARAM

class Config(object):
    """Config -> .."""
    def suit_config(self):
        params = {
            SIZE_PARAM: 32,
            BASE_PATH_PARAM: "",
            INFERENCE_MODEL_DIR_PARAM: "D:\\Imagefy\\resources\\models\\InceptionResNetV2.1\\inference\\1602108545" 
        }
        return params
