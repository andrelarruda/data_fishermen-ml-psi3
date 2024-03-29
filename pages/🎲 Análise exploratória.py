from cProfile import label
import streamlit as st
import pandas as pd
import numpy as np
from utils.data import Data
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(page_title="Stroke Analysis", page_icon=":chart:")


class ExploratoryAnalysis:
    def __init__(self):
        self.data = Data().get_data()
    
    #Dados de AVC para os gráficos BoxPlots
    def get_stroke_data(self):
        return self.data

    #Dados de AVC para os gráficos Distplots
    def get_dist_stroke_data(self, column):
        return self.data.loc[self.data["stroke"] == 1][column]
    def get_dist_nostroke_data(self, column):
        return self.data.loc[self.data["stroke"] == 0][column]

    def distplot(self, hist_data, layoutTitle):
        group_labels = ['Sem AVC', 'Com AVC']
        colors = ['#2BCDC1', '#ff0000']

        fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_curve = True)
        fig.update(layout_title_text = layoutTitle)
        
        st.plotly_chart(fig, use_container_width=True)    
    
    def exploratory_analysis(self):
        st.header("Análise exploratória")
        st.subheader("Descrição")
        st.write(self.data.describe())
        
    def age_box_plot(self):
        st.subheader("Idade")
        st.text_area(label="", value="Abaixo temos um gráfico demográfico do parâmetro idade. É possível perceber que a maioria das idades dos indivíduos participantes da pesquisa está distribuída entre 59 e 78 anos, porém podemos notar ainda a presença de alguns valores discrepantes inferiores. Estes valores representam indivíduos que tiveram AVC com idade de 1 e 14 anos. Embora a ocorrência de AVC em crianças seja rara, é possível. Estudos mostram que o AVC pode acometer até 0,013% de crianças. Portanto os valores listados como outliers permanecem relevantes para o estudo.", height=200)
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data, x = "stroke", y = "age", color = "stroke", color_discrete_sequence = ['#ff0000', '#2BCDC1'], labels={'stroke': 'AVC', 'age': 'Idade'}))

    def age_distplot(self):
        st.subheader('AVC x Idade')
        st.text_area(label="", value="Abaixo temos um gráfico distributivo, tendo como parâmetro a idade dos pacientes saudáveis e os que tiveram a ocorrência de AVC. Fica notório uma maior tendência de casos em pessoas que possuem idade elevada.")
        x1 = self.get_dist_nostroke_data('age')
        x2 = self.get_dist_stroke_data('age')
        hist_data = [x1, x2]

        self.distplot(hist_data, 'Casos de AVC por idade')
    
    def bmi_box_plot(self):
        st.subheader("IMC")
        st.text_area(label="", value="No caso do IMC, a maioria dos valores se concentra entre 27 e 32.5, com a presença de vários outliers, principalmente superiores. Com valor mínimo em 16.9, e valor máximo em 56.6. A mediana é 28.89, aproximadamente. Para os outliers, é possível verificar que são valores possíveis de IMC. Representam pessoas muito acima ou muito abaixo do peso.")
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data, y = "bmi", x = "stroke", color = "stroke", color_discrete_sequence = ['#ff0000', '#2BCDC1'], labels={'stroke': 'AVC', 'bmi': 'Índice de Massa Corporal'}))

    def bmi_distplot(self):
        st.subheader('Distribuição e comparação IMC')
        st.text_area(label="", value="Ao analisar o gráfico da distirbuição de IMC, é perceptível que pessoas sobre-peso não possuem maior tendência a desenvolver AVC. No entanto, como maior parte da amostra encontra-se na região intermediária do gráfico, esse pode ser um dos fatores.")
        x1 = self.get_dist_nostroke_data('bmi')
        x2 = self.get_dist_stroke_data('bmi')
        hist_data = [x1, x2]

        self.distplot(hist_data, 'Gráfico distributivo IMC')

    def avg_glucose_level_box_plot(self):
        st.subheader("Nível de glicose")
        data = self.get_stroke_data()
        st.plotly_chart(px.box(data, y = "avg_glucose_level", x = "stroke", color = "stroke", color_discrete_sequence = ['#ff0000', '#2BCDC1'], labels={'stroke': 'AVC', 'avg_glucose_level': 'Nível médio de glicose' }))

    def glucose_distplot(self):
        st.subheader('Distribuição e comparação Glicose')
        st.text_area(label="", value="Diferente do gráfico que analisa o IMC, o gráfico da glicose mostra uma tendência de casos em pacientes com a glicose elevada.")
        x1 = self.get_dist_nostroke_data('avg_glucose_level')
        x2 = self.get_dist_stroke_data('avg_glucose_level')
        hist_data = [x1, x2]

        self.distplot(hist_data, 'Gráfico distributivo glicose')

    def data_balancing(self):
        st.subheader('Distribuição das Ocorrências de AVC')

        distribution_stroke = self.data.stroke.value_counts()
        st.plotly_chart(px.bar(distribution_stroke, orientation='v', x=['Não teve AVC', 'Teve AVC'], y='stroke', labels={ 'stroke': 'Número de Ocorrências', 'x': 'Categoria', '#2BCDC1': 'Não teve AVC', '#ff0000': 'Teve AVC'}, color={'#2BCDC1': 'Não teve AVC', '#ff0000': 'Teve AVC'} ))

        st.text_area(label="", value="O gráfico de distribuição nos permite ter uma visão geral da distribuição de ocorrências do AVC nos dados. Podemos perceber que existem mais casos que não houveram AVC, o que demonstra que esses dados estão desbalanceados. Portanto, para efetuar a classificação sobre esses dados, será necessário antes efetuar o seu balanceamento.", height=120)

    def scatter_plot(self):
        st.subheader('Gráfico de dispersão para nível médio de glicose')
        st.text_area(label='Nível de Glicose x Idade', value='Utilizando uma amostra de 1000 indivíduos, abaixo temos um gráfico de dispersão (scatter plot) que nos mostra a relação entre o nível médio de glicose e a idade. Aparetemente é uma relação neutra, mas também é possível observar a possível existência de dois grupos distintos, separados pela linha horizontal (eixo y) no valor 150. Podemos notar também que a ocorrência de AVC é mais acentuada à partir dos 40 anos. Estatisticamente, se olharmos o diagrama de caixa em relação ao eixo y, 25% dos indivíduos que tiveram AVC possuem valores de nível médio de glicose abaixo da mediana, entre 79,57 e 105,22. Os outros 25% estão distribuídos acima do valor da mediana, até 196,76.', height=300)

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

        st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, color='stroke', hover_data=['bmi', 'hypertension', 'heart_disease'], labels={ 'avg_glucose_level': 'Nível médio de glicose', 'age': 'Idade', 'stroke': 'AVC', 'bmi': 'IMC', 'hypertension': 'Hipertensão', 'heart_disease': 'Doença cardíaca'}, color_discrete_sequence=['#C72934', '#a2a79e'], marginal_y='box'))

        # IMC - Scatter Plot
        st.text_area(label='IMC x Idade', value='Abaixo plotamos o mapa que relaciona o IMC com a Idade, considerando os casos que houveram ocorrência de AVC e os que não houveram. Os pontos aparentam ser mais uniformes, não permitindo extrair informações relevantes para o estudo. Há alguns pontos discrepantes em relação ao IMC, porém de indivíduos que não tiveram AVC.', height=100)

        x_axis_2 = df['age']
        y_axis_2 = df['bmi']

        st.plotly_chart(px.scatter(df, x=x_axis_2, y=y_axis_2, color='stroke', hover_data=['bmi', 'hypertension', 'heart_disease', "avg_glucose_level"], labels={ 'avg_glucose_level': 'Nível médio de glicose', 'age': 'Idade', 'stroke': 'AVC', 'bmi': 'IMC', 'hypertension': 'Hipertensão', 'heart_disease': 'Doença cardíaca'}, color_discrete_sequence=['#C72934', '#a2a79e']))

        # IMC x Glicose
        st.text_area(label='IMC x Nível médio de glicose', value='Com o seguinte gráfico conseguimos perceber a divisão dos pontos em algo que se assemelha a 2 grupos: um mais à esquerda - com vários pontos de indivíduos sem ocorrência de AVC - e outro mais à direita - tendo como maioria indivíduos que tiveram AVC. Porém, como podemos perceber ao analisar o diagrama de caixa para o nível médio de glicose, 25% dos indivíduos que tiveram AVC registraram glicose abaixo da mediana.', height=200)

        x_axis_3 = df['avg_glucose_level']
        y_axis_3 = df['bmi']

        st.plotly_chart(px.scatter(df, x=x_axis_3, y=y_axis_3, color='stroke', hover_data=['bmi', 'hypertension', 'heart_disease', "avg_glucose_level"], labels={ 'avg_glucose_level': 'Nível médio de glicose', 'age': 'Idade', 'stroke': 'AVC', 'bmi': 'IMC', 'hypertension': 'Hipertensão', 'heart_disease': 'Doença cardíaca'}, color_discrete_sequence=['#C72934', '#a2a79e'], marginal_x="box"))

exploratory_analysis_page = ExploratoryAnalysis()
exploratory_analysis_page.exploratory_analysis()
exploratory_analysis_page.data_balancing()
exploratory_analysis_page.age_box_plot()
exploratory_analysis_page.age_distplot()
exploratory_analysis_page.bmi_box_plot()
exploratory_analysis_page.bmi_distplot()
exploratory_analysis_page.avg_glucose_level_box_plot()
exploratory_analysis_page.glucose_distplot()
exploratory_analysis_page.scatter_plot()