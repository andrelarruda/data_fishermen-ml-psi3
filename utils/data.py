import pandas as pd
import streamlit as st

class Data:

    def __init__(self):
        self.DATA_URL = ('data/newstroke-data.csv')
    
    def get_data(self):
        data = pd.read_csv(self.DATA_URL)
        data['age'] = data['age'].astype(int)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        del data['id']
        return data

    def get_age_data(slef):
        return self.get_data

