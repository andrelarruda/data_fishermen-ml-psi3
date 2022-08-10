import streamlit as st
import pandas as pd
import numpy as np
from utils.data import Data

st.set_page_config(page_title="Stroke Analysis", page_icon=":broken_heart:")


class ExploratoryAnalysis:
    def __init__(self):
        self.data = Data().get_data()
    
    def exploratory_analysis(self):
        st.subheader("Exploratory data Analysis")
        st.write(self.data.describe())

exploratory_analysis_page = ExploratoryAnalysis()
exploratory_analysis_page.exploratory_analysis()