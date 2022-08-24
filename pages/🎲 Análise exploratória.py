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
        st.subheader("Nível de glicose")
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data,y="avg_glucose_level",x="stroke"))

    def data_balancing(self):
        st.subheader('Distribução das Ocorrências de AVC')

        distribution_stroke = self.data.stroke.value_counts()
        st.plotly_chart(px.bar(distribution_stroke, orientation='v', x=['Não teve AVC', 'Teve AVC'], y='stroke', labels={ 'stroke': 'Número de Ocorrências', 'x': 'Categoria'} ))

        st.text_area(label="", value="O gráfico de distribuição nos permite ter uma visão geral da distribuição de ocorrências do AVC nos dados. Podemos perceber que existem mais casos que não houveram AVC, o que demonstra que esses dados estão desbalanceados. Portanto, para efetuar a classificação sobre esses dados, será necessário antes efetuar o seu balanceamento.", height=120)

    def scatter_plot(self):
        st.subheader('Mapa de espalhamento para nível médio de glicose')
        st.text_area(label='Nível de Glicose x Idade', value='Utilizando uma amostra de 1000 indivíduos, abaixo temos um gráfico de espalhamento (scatter plot) que nos mostra a relação entre o nível médio de glicose e a idade. É possível notar que o nível médio de glicose aparentemente não é um fator de risco para a ocorrência do AVC. Conforme observamos no subgráfico de distribuição das idades, os respectivos valores estão distribuídos de maneira diferente, para os indivíduos que tiveram AVC e os que não tiveram. Podemos notar também que a ocorrência de AVC é mais acentuada à partir dos 40 anos. Estatisticamente, se olharmos o box plot em relação ao eixo y, 50%% dos indivíduos que tiveram AVC possuem valores de nível médio de glicose abaixo da mediana, que é de 105.22. Os outros 50%% estão distribuídos acima do valor da mediana. O que acaba aumentando ainda mais nossa suspeita de que o nível médio de glicose não é um fator determinante para a ocorrência de AVC. A idade, por outro lado, aparenta ser um parâmetro importante para ocorrência do AVC, uma vez que a maioria dos valores estão distribuídos acima dos 59 anos (analisando o box plot em relação ao eixo x).', height=300)

        q = 1000
        df = self.data.head(q)

        df['gender'] = df['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
        df['residence_type'] = df['residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)
        df['work_type'] = df['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':3,'Never_worked':4}).astype(np.uint8)
        df['smoking_status'] = df['smoking_status'].replace({'smokes':1,'never smoked':0,'formerly smoked':2 ,'Unknown':3}).astype(np.uint8)
        df['ever_married'] = df['ever_married'].replace({'Yes':1,'No':0}).astype(np.uint8)
        df['stroke'] = df['stroke'].replace({'1':1,'0':0}).astype(np.uint8)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        x_axis = df['age']
        y_axis = df['avg_glucose_level']

        st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, color='stroke', hover_data=['bmi', 'hypertension', 'heart_disease'], labels={ 'avg_glucose_level': 'Nível médio de glicose', 'age': 'Idade', 'stroke': 'AVC', 'bmi': 'IMC', 'hypertension': 'Hipertensão', 'heart_disease': 'Doença cardíaca'}, color_discrete_sequence=['#C72934', '#a2a79e'], marginal_x="box", marginal_y="box"))

        # IMC - Scatter Plot
        st.text_area(label='IMC x Idade', value='Abaixo plotamos o mapa que relaciona o IMC com a Idade, considerando os casos que houveram ocorrência de AVC e os que não houveram. Os pontos aparentam ser mais uniformes, não permitindo extrair informações relevantes.', height=100)

        x_axis_2 = df['age']
        y_axis_2 = df['bmi']

        st.plotly_chart(px.scatter(df, x=x_axis_2, y=y_axis_2, color='stroke', hover_data=['bmi', 'hypertension', 'heart_disease', "avg_glucose_level"], labels={ 'avg_glucose_level': 'Nível médio de glicose', 'age': 'Idade', 'stroke': 'AVC', 'bmi': 'IMC', 'hypertension': 'Hipertensão', 'heart_disease': 'Doença cardíaca'}, color_discrete_sequence=['#C72934', '#a2a79e']))

        # IMC x Glicose
        st.text_area(label='IMC x Nível médio de glicose', value='Com o seguinte gráfico conseguimos perceber a divisão dos pontos em algo que se assemelha a 2 clusteres: um mais à esquerda - com vários pontos de indivíduos sem ocorrência de AVC - e outro mais à direita - tendo como maioria indivíduos que tiveram AVC. Porém, como podemos perceber ao analisar o box plot para o nível médio de glicose, 50%% dos indivíduos que tiveram AVC registraram glicose abaixo da mediana. Mais uma vez fortalecendo a tese de que esse parâmetro não tem relação direta com a ocorrência de AVC, porém nenhuma hipótese ainda foi descartada, uma vez que tal cenário pode ser em decorrência do desbalanceamento dos dados.', height=200)

        x_axis_3 = df['avg_glucose_level']
        y_axis_3 = df['bmi']

        st.plotly_chart(px.scatter(df, x=x_axis_3, y=y_axis_3, color='stroke', hover_data=['bmi', 'hypertension', 'heart_disease', "avg_glucose_level"], labels={ 'avg_glucose_level': 'Nível médio de glicose', 'age': 'Idade', 'stroke': 'AVC', 'bmi': 'IMC', 'hypertension': 'Hipertensão', 'heart_disease': 'Doença cardíaca'}, color_discrete_sequence=['#C72934', '#a2a79e'], marginal_x="box"))

exploratory_analysis_page = ExploratoryAnalysis()
exploratory_analysis_page.exploratory_analysis()
exploratory_analysis_page.data_balancing()
exploratory_analysis_page.age_box_plot()
exploratory_analysis_page.bmi_box_plot()
exploratory_analysis_page.avg_glucose_level_box_plot()
exploratory_analysis_page.scatter_plot()