import streamlit as st
import pandas as pd
import numpy as np

st.title('Strokes')

AGE_COLUMN = 'age'
DATA_URL = ('./data/stroke-data.csv')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

data = load_data(10000)

st.subheader('Raw data')
st.write(data)

df_stroke=data.loc[data["stroke"]==1]
df_no_stroke=data.loc[data["stroke"]==1]

st.subheader('Age')

df_stroke_age = df_stroke.groupby("age")["stroke"].count()
st.area_chart(data=df_stroke_age)

st.subheader('Residence Type')

df_stroke_glucose = df_stroke.groupby("residence_type")["stroke"].count()
st.bar_chart(data=df_stroke_glucose)
