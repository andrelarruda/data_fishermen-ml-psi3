import streamlit as st
import pandas as pd
import numpy as np
from utils.data import Data
import plotly.express as px


st.set_page_config(page_title="Stroke Analysis", page_icon=":broken_heart:")


class ExploratoryAnalysis:
    def __init__(self):
        self.data = Data().get_data()

    def get_stroke_data(self):
        return self.data.loc[self.data["stroke"]==1]

    
    def exploratory_analysis(self):
        st.header("Análise exploratória")
        st.subheader("Descrição")
        st.write(self.data.describe())

    def age_box_plot(self):
        st.subheader("Idade")
        st.text_area(label="", value="Abaixo temos um gráfico demográfico do parâmetro idade. É possível perceber que a maioria das idades dos indivíduos participantes da pesquisa está distribuída entre 59 e 78 anos, porém podemos notar ainda a presença de alguns valores discrepantes, superiores e inferiores.")
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data,y="age",x="stroke"))
    
    def bmi_box_plot(self):
        st.subheader("IMC")
        st.text_area(label="", value="No caso do IMC, a maioria dos valores se concentra entre 27 e 32.5, com a presença de vários outliers. Com valor mínimo em 16.9, e valor máximo em 56.6. A mediana é 28.89, aproximadamente.")
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data,y="bmi",x="stroke"))

    def avg_glucose_level_box_plot(self):
        st.subheader("Nível de glucose")
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data,y="avg_glucose_level",x="stroke"))


exploratory_analysis_page = ExploratoryAnalysis()
exploratory_analysis_page.exploratory_analysis()
exploratory_analysis_page.age_box_plot()
exploratory_analysis_page.bmi_box_plot()
exploratory_analysis_page.avg_glucose_level_box_plot()