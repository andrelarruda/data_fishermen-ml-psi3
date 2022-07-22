import streamlit as st
import pandas as pd
import numpy as np

st.title('Strokes')

AGE_COLUMN = 'age'
DATA_URL = ('./data/stroke-data.csv')

@st.cache(allow_output_mutation=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data['age'] = data['age'].astype(int)
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
age_selection = st.sidebar.slider('Idade', 0, int(data['age'].max(axis=0)), value=(0, int(data['age'].max(axis=0)+1)))
minimum_age = age_selection[0]
maximum_age = age_selection[1] + 1

data = data[((data['age'] >= minimum_age) & (data['age'] <= maximum_age))]
#  Fim Filtro idade

#  Filtro IMC
bmi_selection = st.sidebar.slider('IMC', 0, int(data['bmi'].max(axis=0)), value=(0, int(data['bmi'].max(axis=0))))
minimum_bmi = bmi_selection[0]
maximum_bmi = bmi_selection[1]

data = data[((data['bmi'] >= minimum_bmi) & (data['bmi'] <= maximum_bmi))]
# Fim Filtro IMC

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


# Exibir tabela
st.subheader('Raw data')
st.write(data)
df_stroke=data.loc[data["stroke"]==1]
df_no_stroke=data.loc[data["stroke"]==1]
# Fim exibir tabela

# Grafico exibir AVC por Idade
st.subheader('Age')
df_stroke_age = df_stroke.groupby("age")["stroke"].count()
st.area_chart(data=df_stroke_age)
# Fim grafico Exibir AVC por Idade

# Grafico AVC por tipo de localidade
st.subheader('Residence Type')
df_stroke_glucose = df_stroke.groupby("residence_type")["stroke"].count()
st.bar_chart(data=df_stroke_glucose)
# Fim grafico AVC por tipo de localidade

# Grafico numero de AVC por Smoking_status
st.subheader('Smoker')
df_stroke_smoker = df_stroke.groupby("smoking_status")["stroke"].count()
st.bar_chart(data=df_stroke_smoker)
# Fim grafico numero de AVC por Smoking_status