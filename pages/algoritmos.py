import pandas as pd
import json
from utils.data import Data
import streamlit as st
import numpy as np
import plotly.express as px
from utils.data import Data
import matplotlib.pyplot as plt
import matplotlib
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import plot_confusion_matrix, classification_report,confusion_matrix,f1_score, classification_report

class Algorithms:
    def __init__(self):
        self.data = Data().get_data()
        self.balanced_data = self.get_balanced_dataframe()

    def get_updated_data(self):
        df = pd.DataFrame(self.data)

        colunas = [coluna for coluna in df.columns if coluna not in ['id','stroke']]

        conts = ['age','avg_glucose_level','bmi']

        #Alterando as variáveis do tipo str para numéricos
        df['gender'] = df['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
        df['residence_type'] = df['residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)
        df['work_type'] = df['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)
        df['smoking_status'] = df['smoking_status'].replace({'smokes':1,'never smoked':0,'formerly smoked':2 ,'Unknown':-1}).astype(np.uint8)
        df['ever_married'] = df['ever_married'].replace({'Yes':1,'No':0}).astype(np.uint8)

        return df

    def get_train_and_test_data(self):
        self.data = self.get_updated_data()
        # Definindo os conjuntos de dados de entrada e saida
        self.x = self.data[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']]
        self.y = self.data['stroke']

        # obtendo os dados de treino e teste
        return train_test_split(self.x, self.y, train_size=0.3, random_state=42)

    def get_balanced_dataframe(self):
        x_train, self.x_test, y_train, self.y_test = self.get_train_and_test_data()
        
        #Balanceando os dados com Smote (oversample)
        oversample = SMOTE()
        X_train_resh, y_train_resh = oversample.fit_resample(x_train, y_train.ravel())
        self.x_train_resampled = X_train_resh
        self.y_train_resampled = y_train_resh
        
        #criando um novo dataframe com os dados balanceados
        balanced_dataframe = pd.DataFrame(y_train_resh,columns=['stroke'])
        return balanced_dataframe

    def plot_imbalanced_distribution_chart(self):
        before_balancing=self.data.groupby("stroke")["stroke"].count().reset_index(name='qtd_stroke')

        before_balancing['stroke'] = before_balancing['stroke'].replace({1:'Sim',0:'Não'}).astype(str)

        st.text_area(label='Dados não balanceados.', value='Observando o gráfico abaixo podemos notar que o número de casos confirmados de AVC é muito menor do que os casos não confirmados, dessa maneira seria difícil encontrar um modelo que trouxesse um resultado satisfatório.', height=100)

        fig = px.bar(before_balancing, x="stroke", y="qtd_stroke", color="stroke", title="Antes do balanceamento", labels={ 'qtd_stroke': 'Quantidade de ocorrências', 'stroke': 'Ocorrência de AVC', 'no_stroke': 'Não', 'yes_stroke': 'Sim' })
        st.plotly_chart(fig,use_container_width=True)

    def plot_balanced_distribution_chart(self):
        after_balancing=self.balanced_data.groupby("stroke")["stroke"].count().reset_index(name='qtd_stroke')

        after_balancing['stroke'] = after_balancing['stroke'].replace({1:'Sim',0:'Não'}).astype(str)

        st.text_area(label='Dados  balanceados.', value='Foi utilizado o SMOTE(Synthetic Minority Oversampling Technique) para balancear os dados e assim obter melhores resultados. Como o número de casos positivos é muito menor  do que os de casos negativos, o SMOTE foi ideal para esse balanceamento, já que ele ira preencher com mais casos positivos  nosso dataframe', height=150)

        fig = px.bar(after_balancing, x="stroke", y="qtd_stroke", color="stroke", title="Depois do balanceamento", labels={ 'qtd_stroke': 'Quantidade de ocorrências', 'stroke': 'Ocorrência de AVC', 'no_stroke': 'Não', 'yes_stroke': 'Sim' })
        st.plotly_chart(fig,use_container_width=True)

    # Função para gerar matrizes de confusão
    def matrix(self, classifier, classifierName, score):
        #score = score.mean()
        st.subheader('Matriz de Confusão ' + classifierName + ':')
        #st.text_area(label = "Mean f1 score:", value = classifierName + " mean: " + str(score),  height = 1)
        score.pop('accuracy')
        st.table(score)
        fig = px.imshow(classifier, text_auto=True, aspect="auto", color_continuous_scale='ylgnbu',
                    labels=dict(x="Valores previstos ", y="Valores reais", color="Número de casos"),
                    x=['Predição negativa', 'Predição positiva'],
                    y=['Negativo', 'Positivo']
                )

        fig.update_xaxes(side="bottom")
        st.plotly_chart(fig)

    def rf_feat_importance(self, m, df):
        return pd.DataFrame({'Feature' : df.columns, 'Importance' : m.feature_importances_}).sort_values('Importance', ascending=False)

    def calculate_score_random_forest(self):
        self.rf_pipeline = Pipeline(steps = [('scale',StandardScaler()),('RF',RandomForestClassifier(random_state=42))])

        self.rf_pipeline.fit(self.x_train_resampled, self.y_train_resampled)
        predictionsRF = self.rf_pipeline.predict(self.x_test)
        rfcm = confusion_matrix(self.y_test, predictionsRF)
        dict_rf = classification_report(self.y_test,predictionsRF,output_dict=True)
        self.matrix(rfcm, 'Random Forest', dict_rf)

    def calculate_score_logistic_regression(self):
        logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('LR',LogisticRegression(random_state=42))])
        logreg_pipeline.fit(self.x_train_resampled, self.y_train_resampled)
        predictionsLR = logreg_pipeline.predict(self.x_test)
        lgrmc = confusion_matrix(self.y_test, predictionsLR)
        dict_lr = classification_report(self.y_test,predictionsLR,output_dict=True)

        self.matrix(lgrmc, 'Logistic Regression', dict_lr)

    def calculate_score_svm(self):
        svm_pipeline = Pipeline(steps = [('scale',StandardScaler()),('SVM',SVC(random_state=42))])
        svm_pipeline.fit(self.x_train_resampled, self.y_train_resampled)
        predictionsSVC=svm_pipeline.predict(self.x_test)
        svmcm=confusion_matrix(self.y_test, predictionsSVC)
        dict_svc = classification_report(self.y_test,predictionsSVC,output_dict=True)

        self.matrix(svmcm, 'Support Vector Machines', dict_svc)

    def show_feature_importance_random_forest(self):
        #classificador
        colors = ["lightgray","lightgray","#0f4c81"]
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        #Obtendo a Feature importance do Random Forest
        fi_random_florest = self.rf_feat_importance(self.rf_pipeline['RF'], self.x)

        fig = px.bar(fi_random_florest, y='Importance', x='Feature',
                    title="Feature Importance", labels={ 'Importance': 'Importância' })
        st.plotly_chart(fig)

algorithms_page = Algorithms()
algorithms_page.plot_imbalanced_distribution_chart()
algorithms_page.plot_balanced_distribution_chart()
algorithms_page.calculate_score_random_forest()
algorithms_page.calculate_score_logistic_regression()
algorithms_page.calculate_score_svm()
algorithms_page.show_feature_importance_random_forest()


