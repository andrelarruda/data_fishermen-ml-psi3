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

#  Sidebar
st.sidebar.header('Filtros')

#  **FILTROS**
# Filtro de ocorrencia de AVC
avc_ocurrence_option = st.sidebar.radio(
     "Ocorrência de AVC",
     ('Todos', 'Sim', 'Não'))

if avc_ocurrence_option == 'Sim':
    data = data[data['stroke'] == 1]
elif avc_ocurrence_option == 'Não':
    data = data[data['stroke'] == 0]
#  Fim Filtro de ocorrencia de AVC

#  Filtro idade
age_selection = st.sidebar.slider('Idade', 0, int(data['age'].max(axis=0)), value=(0, int(data['age'].max(axis=0))))
minimum_age = age_selection[0]
maximum_age = age_selection[1]

data = data[((data['age'] >= minimum_age) & (data['age'] <= maximum_age))]
#  Fim Filtro idade

# Filtro de tipo de trabalho
work_options = ['Todos'] + list(data['work_type'].unique())
work_type_select = st.sidebar.selectbox(
     "Modo de trabalho", options=work_options)

if work_type_select == 'Private':
    data = data[data['work_type'] == 'Private']
elif work_type_select == 'Self-employed':
    data = data[data['work_type'] == 'Self-employed']
elif work_type_select == 'Govt-job':
    data = data[data['work_type'] == 'Govt-job']
elif work_type_select == 'children':
    data = data[data['work_type'] == 'children']
elif work_type_select == 'Never_worked':
    data = data[data['work_type'] == 'Never_worked']
else:
    data = data
# exemplo: https://discuss.streamlit.io/t/filtering-data-with-pandas/20724
#  Fim Filtro de tipo de trabalho

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
