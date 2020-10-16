"""
@Author: Yuval Kaneti 
"""
import streamlit as st
# Normal stuff
LOGO_PATH = "imagefy\\www\\assets\\logo.png"
TITLE = "Imagefy"
HEADER = "*The Photographer's Best Friend*"
LINE_BREAK = "---"
BETA_TESTING = "*This Website is in its Very First Stages, Please Rate you experience at the `sidebar on your left`*"

# Sidebar & selectbox
WHAT_IS_IT_PARAM = "WHAT IS IT"
WHAT_IS_IT_VALUE = "*Imagefy is a website designed to save a busy `Photographer` time by picking for him the `best images`*"
HOW_DOES_IT_WORK_PARAM = "HOW DOES IT WORK"
HOW_DOES_IT_WORK_VALUE = "*Imagefy has seen `Hundreds of Thousands` of images and their online ratings, it uses `machine learning` to predict the `score` of your images and only shows you the best ones*"
GET_STARTED_PARAM = "GET STARTED"
GET_STARTED_VALUE = "*All you nead to do is to choose about ~50 images press the `Browse files` button & press `Start Imagefy`*"
WHO_AM_I_PARAM = "WHO AM I"
WHO_AM_I_VALUE = "*I'm an 19 years old guy from Israel and an Enthusiast Photgrapher who's tired of doing things the manual way*"
RATE_SLIDER_LABEL = "Rate your Experience"
SIDEBAR_OPTIONS_TITLE = 'Tell me what to show you'
SIDEBAR_OPTIONS_DICT = {
    WHAT_IS_IT_PARAM: WHAT_IS_IT_VALUE,
    HOW_DOES_IT_WORK_PARAM: HOW_DOES_IT_WORK_VALUE,
    GET_STARTED_PARAM: GET_STARTED_VALUE,
    WHO_AM_I_PARAM: WHO_AM_I_VALUE,
}

# Upload file
UPLOAD_LABEL = "Upload Images For Imagefy."
UPLOAD_FILE_TYPES = ['png', 'jpg']

# Top Images Slider
TOP_K_SLIDER_LABEL = "How many Images to show"

# Start button
START_IMAGEFY_BUTTON = 'Start Imagefy'

# Extender
SEE_EXPLANATION = "See explanation"
UPLOAD_EXPLANATION = "*You should choose about ~50 images you would like imagefy to rate*"
RUN_EXPLANATION = f"*After pressing `{START_IMAGEFY_BUTTON}` Imagefy will go through the images, rate them and only return the best ones, \
    You Can control The Amount of images Imagefy returns with the `slider` above*"

PROGRESS_0 = 0
PROGRESS_10 = 10
PROGRESS_25 = 25
PROGRESS_50 = 50
PROGRESS_75 = 75
PROGRESS_100 = 100