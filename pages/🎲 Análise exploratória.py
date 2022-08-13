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
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data,y="age",x="stroke"))
    
    def bmi_box_plot(self):
        st.subheader("IMC")
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