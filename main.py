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

data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Data loaded')

st.subheader('Raw data')
st.write(data)

#st.subheader('Quantity of null data by column')
#st.text(df.isnull().sum())

x = data[['age', 'hypertension', 'gender']].copy()

df_stroke=data.loc[data["stroke"]==1]
df_no_stroke=data.loc[data["stroke"]==1]

st.subheader('Age')

df_stroke_age = df_stroke.groupby("age")["stroke"].count()
st.area_chart(data=df_stroke_age)

st.subheader('Residence Type')

df_stroke_glucose = df_stroke.groupby("residence_type")["stroke"].count()
st.bar_chart(data=df_stroke_glucose)
