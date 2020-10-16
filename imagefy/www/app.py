"""
@Author: Yuval Kaneti 
"""

import time
import logging; logger = logging.getLogger('Imagefy')
import streamlit as st; st.beta_set_page_config(page_title="Imagefy", page_icon="imagefy\\www\\assets\\logo.png", layout='centered', initial_sidebar_state='expanded')
try:
    from imagefy.suits.inception_suit import InceptionSuit
    from imagefy.utils.config import Config
    from imagefy.utils.common import IMAGES_PARAM, TOP_PARAM, PROGRESS_BAR_PARAM
    from imagefy.utils.web_utils import file_uploader, top_k_slider, run_button, messages_placeholder, side_bar, progress_bar, show_images
    from imagefy.www.common import PROGRESS_100, HEADER, TITLE, LINE_BREAK, BETA_TESTING
except ImportError as err:
    logger.error(err, exc_info=True)

class App(object):
    """App -> The Streamlit Imagefy App."""
    def __init__(self): 
        st.markdown(TITLE)
        st.header(HEADER)
        st.markdown(LINE_BREAK) 
        st.info(BETA_TESTING)    
        st.sidebar.title(TITLE)
        self.page_load()
        self.config = Config()

    def page_load(self):
        self.side_bar = side_bar()
        self.file_upload_dailog = file_uploader()
        self.top_k_sld = top_k_slider()
        self.run_btn = run_button()
        self.msg_placeholder = messages_placeholder()

    def run(self):
        try:
            self.msg_placeholder.empty()
            self.files = self.file_upload()

            if self.run_btn and self.files is None:
                st.warning("*Please Select Files befor Starting*")
                st.stop()

            if self.files is not None and self.run_btn:
                start = time.time()
                self.msg_placeholder.info("*This might take some time, Depending on how may images you have chosen*")
                self.progress_bar = progress_bar()
                suit_params = self._update_suit_params()                
                self.suit = InceptionSuit(**suit_params)
                images = self.suit.run()          
                st_images = []
                st_captions = []
                for image in images:
                    if image.top:
                        st_images.append(image.data)
                        st_captions.append(f"{image.file_name}")

                self.progress_bar.progress(PROGRESS_100)
                # st.image(st_images, width=150, caption=st_captions)
                show_images(st_images, st_captions)
                end = round(time.time() - start)
                self.msg_placeholder.info(f"It took Imagefy {end} Seconds to go through {len(images)} Images!")
                st.balloons()

        except Exception as err:
            logger.error(err, exc_info=True)

    def _update_suit_params(self):
        suit_params = self.config.suit_config()
        suit_params.update({IMAGES_PARAM: self.files, TOP_PARAM: self.top_k_sld, PROGRESS_BAR_PARAM: self.progress_bar})
        return suit_params
    
    def file_upload(self):
        files = self.file_upload_dailog
        if len(files) > 0:
            return files
        else:
            return None      

    @st.cache
    def local_css(self, file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if __name__ == '__main__':
    App().run()