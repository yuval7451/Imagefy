"""
@Author: Yuval Kaneti 
"""
import logging; logger = logging.getLogger('Imagefy')
import streamlit as st
from base64 import b64encode
from imagefy.www.common import  TOP_K_SLIDER_LABEL, START_IMAGEFY_BUTTON, \
    SIDEBAR_OPTIONS_DICT, UPLOAD_FILE_TYPES, UPLOAD_LABEL, WHAT_IS_IT_PARAM, \
    HOW_DOES_IT_WORK_PARAM, GET_STARTED_PARAM, RUN_EXPLANATION, RATE_SLIDER_LABEL, \
    SIDEBAR_OPTIONS_TITLE, WHO_AM_I_PARAM, SEE_EXPLANATION, UPLOAD_EXPLANATION, PROGRESS_0


@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=False)
def side_bar():
    """
    """
    sidebar_selectbox = st.sidebar.selectbox(SIDEBAR_OPTIONS_TITLE, (WHAT_IS_IT_PARAM, HOW_DOES_IT_WORK_PARAM, GET_STARTED_PARAM, WHO_AM_I_PARAM))
    st.sidebar.write(SIDEBAR_OPTIONS_DICT[sidebar_selectbox])
    rating = st.sidebar.slider(label=RATE_SLIDER_LABEL, min_value=0, max_value=10, value=0, step=1)
    if rating != 0:
        logger.info(f"RATING: {rating}")

def file_uploader():
    """
    """
    file_upload_dailog = st.file_uploader(label=UPLOAD_LABEL, type=UPLOAD_FILE_TYPES, accept_multiple_files=True)
    with st.beta_expander(SEE_EXPLANATION):
        st.markdown(UPLOAD_EXPLANATION)

    return file_upload_dailog

@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=False)
def top_k_slider():
    """
    """
    return st.slider(label=TOP_K_SLIDER_LABEL, min_value=1, max_value=100, value=10, step=1)

@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=False)
def run_button():
    """
    """
    run_btn = st.button(START_IMAGEFY_BUTTON)
    with st.beta_expander(SEE_EXPLANATION):
        st.markdown(RUN_EXPLANATION)

    return run_btn

@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=False)
def progress_bar():
    """
    """
    return st.progress(PROGRESS_0)

@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=False)
def messages_placeholder():
    """
    """
    return st.empty()

def warning_cleanup(messege):
    st.warning(messege)
    st.empty()
    st.stop()

def file_download(image):
    b64 = b64encode(image.data).decode()
    href = f'#### *<a download={image.file_name} href="data:image/png;base64,{b64}">Download {image.file_name}</a>*'
    return href

@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=False)
def show_images(st_images, st_captions):
    st.image(st_images, width=150, caption=st_captions)
