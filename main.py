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
df = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Data loaded')

st.subheader('Raw data')
st.write(df)

#st.subheader('Quantity of null data by column')
#st.text(df.isnull().sum())

x = df[['age', 'hypertension', 'gender']].copy()

#st.table(df['age'].loc[df['stroke']==1])
had_stroke = df['age'].loc[df['stroke']==1]
no_stroke = df['age'].loc[df['stroke']==0]


st.line_chart(data=had_stroke, width=0, height=0, use_container_width=True)


st.subheader('Age')
