from cProfile import label
import streamlit as st
import pandas as pd
import numpy as np
from utils.data import Data
import plotly.express as px


st.set_page_config(page_title="Stroke Analysis", page_icon=":chart:")


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
        st.text_area(label="", value="Abaixo temos um gráfico demográfico do parâmetro idade. É possível perceber que a maioria das idades dos indivíduos participantes da pesquisa está distribuída entre 59 e 78 anos, porém podemos notar ainda a presença de alguns valores discrepantes inferiores.")
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data,y="age",x="stroke"))
    
    def bmi_box_plot(self):
        st.subheader("IMC")
        st.text_area(label="", value="No caso do IMC, a maioria dos valores se concentra entre 27 e 32.5, com a presença de vários outliers, principalmente superiores. Com valor mínimo em 16.9, e valor máximo em 56.6. A mediana é 28.89, aproximadamente.")
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data,y="bmi",x="stroke"))

    def avg_glucose_level_box_plot(self):
        st.subheader("Nível de glucose")
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data,y="avg_glucose_level",x="stroke"))

    def data_balancing(self):
        st.subheader('Distribução das Ocorrências de AVC')

        distribution_stroke = self.data.stroke.value_counts()
        st.plotly_chart(px.bar(distribution_stroke, orientation='v', x=['Não teve AVC', 'Teve AVC'], y='stroke', labels={ 'stroke': 'Número de Ocorrências', 'x': 'Categoria'} ))

        st.text_area(label="", value="O gráfico de distribuição nos permite ter uma visão geral da distribuição de ocorrências do AVC nos dados. Podemos perceber que existem mais casos que não houveram AVC, o que demonstra que esses dados estão desbalanceados. Portanto, para efetuar a classificação sobre esses dados, será necessário antes efetuar o seu balanceamento.", height=120)


exploratory_analysis_page = ExploratoryAnalysis()
exploratory_analysis_page.exploratory_analysis()
exploratory_analysis_page.data_balancing()
exploratory_analysis_page.age_box_plot()
exploratory_analysis_page.bmi_box_plot()
exploratory_analysis_page.avg_glucose_level_box_plot()