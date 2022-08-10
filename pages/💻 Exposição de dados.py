import streamlit as st
import pandas as pd
import numpy as np
from utils.data import Data

st.set_page_config(page_title="Stroke Analysis", page_icon=":broken_heart:")


class HomePage:
    AGE_COLUMN = 'age'
    def __init__(self):
        self.data = Data().get_data()
        self.set_filters()


    def set_filters(self):
        st.sidebar.header('Filtros')
        self.stroke_occurrence_filter()
        self.age_filter()
        self.bmi_filter()
        self.work_type_filter()

    def stroke_occurrence_filter(self):
        avc_occurrence_option = st.sidebar.radio(
            "Ocorrência de AVC",
            ('Todos', 'Sim', 'Não'))

        if avc_occurrence_option == 'Sim':
            self.self.data = self.data[self.data['stroke'] == 1]
        elif avc_occurrence_option == 'Não':
            self.self.data = self.data[self.data['stroke'] == 0]

    def age_filter(self):
        age_selection = st.sidebar.slider('Idade', 0, int(self.data['age'].max(axis=0)), value=(0, int(self.data['age'].max(axis=0)+1)))
        minimum_age = age_selection[0]
        maximum_age = age_selection[1] + 1

        self.data = self.data[((self.data['age'] >= minimum_age) & (self.data['age'] <= maximum_age))]

    def bmi_filter(self):
        bmi_selection = st.sidebar.slider('IMC', 0, int(self.data['bmi'].max(axis=0)), value=(0, int(self.data['bmi'].max(axis=0))))
        minimum_bmi = bmi_selection[0]
        maximum_bmi = bmi_selection[1]

        self.data = self.data[((self.data['bmi'] >= minimum_bmi) & (self.data['bmi'] <= maximum_bmi))]

    def work_type_filter(self):
        work_options = ['Todos'] + list(self.data['work_type'].unique())
        work_type_select = st.sidebar.selectbox(
            "Modo de trabalho", options=work_options)

        if work_type_select == 'Private':
            self.data = self.data[self.data['work_type'] == 'Private']
        elif work_type_select == 'Self-employed':
            self.data = self.data[self.data['work_type'] == 'Self-employed']
        elif work_type_select == 'Govt-job':
            self.data = self.data[self.data['work_type'] == 'Govt-job']
        elif work_type_select == 'children':
            self.data = self.data[self.data['work_type'] == 'children']
        elif work_type_select == 'Never_worked':
            self.data = self.data[self.data['work_type'] == 'Never_worked']

        else:
            self.data = self.data

    def gender_filter(self):                
        gender_options = ['Todos'] + list(self.data['gender'].unique())
        gender_type_select = st.sidebar.selectbox(
            "Gênero", options=gender_options)

        if gender_type_select == 'Male':
            self.data = self.data[self.data['gender'] == 'Male']

        elif  gender_type_select == 'Female':
            self.data = self.data[self.data['gender'] == 'Female']

        elif  gender_type_select == 'Other':
            self.data = self.data[self.data['gender'] == 'Other']
        else:
            self.data = self.data

    def get_stroke_data(self):
        return self.data.loc[self.data["stroke"]==1]

    def raw_data_table(self):
        st.subheader('Raw data')
        st.write(self.data)
        df_stroke=self.data.loc[self.data["stroke"]==1]
        df_no_stroke=self.data.loc[self.data["stroke"]==1]

    def stroke_by_age_graphic(self):
        st.subheader('Age')
        df_stroke_age = self.get_stroke_data().groupby("age")["stroke"].count()
        st.area_chart(data=df_stroke_age)

    def stroke_by_residence_type_graphic(self):
        st.subheader('Residence Type')
        df_stroke_glucose = self.get_stroke_data().groupby("residence_type")["stroke"].count()
        st.bar_chart(data=df_stroke_glucose)
    
    def stroke_by_smoker_graphic(self):
        st.subheader('Smoker')
        df_stroke_smoker = self.get_stroke_data().groupby("smoking_status")["stroke"].count()
        st.bar_chart(data=df_stroke_smoker)

    def stroke_by_gender(self):
        st.subheader('Vítimas por gênero')
        df_gender_victim = self.get_stroke_data().groupby("gender")["stroke"].count()
        st.bar_chart(data=df_gender_victim)


home = HomePage()
home.raw_data_table()
home.stroke_by_age_graphic()
home.stroke_by_residence_type_graphic()
home.stroke_by_smoker_graphic()
home.stroke_by_gender()
