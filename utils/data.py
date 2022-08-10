import pandas as pd
import streamlit as st

class Data:

    def __init__(self):
        self.DATA_URL = ('data/newstroke-data.csv')
    
    @st.cache(allow_output_mutation=True)
    def get_data(self):
        data = pd.read_csv(self.DATA_URL)
        data['age'] = data['age'].astype(int)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data

