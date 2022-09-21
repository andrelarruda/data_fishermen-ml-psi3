from audioop import cross
import pandas as pd
import json
from utils.data import Data
import streamlit as st
import numpy as np
from numpy import mean
from numpy import std
import plotly.express as px
from utils.data import Data
import matplotlib.pyplot as plt
import matplotlib
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import plot_confusion_matrix, classification_report,confusion_matrix,f1_score, classification_report
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
import eli5
from mlxtend.evaluate import paired_ttest_5x2cv

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
    def get_rf_metrics(self):

        rf = RandomForestClassifier(max_features=2,n_estimators=100,bootstrap=True)

        rf.fit(self.x_train_resampled,self.y_train_resampled)

        rfc_tuned_pred = rf.predict(self.x_test)

        print(classification_report(self.y_test,rfc_tuned_pred))

        print('Accuracy Score: ',accuracy_score(self.y_test,rfc_tuned_pred))
        print('F1 Score: ',f1_score(self.y_test,rfc_tuned_pred))
    
    def get_lr_metrics(self):

        

        logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('LR',LogisticRegression(C=0.1,penalty='l2',random_state=42))])

        logreg_pipeline.fit(self.x_train_resampled,self.y_train_resampled)

        logreg_tuned_pred   = logreg_pipeline.predict(self.x_test)

        print(classification_report(self.y_test,logreg_tuned_pred))
        print('Accuracy Score: ',accuracy_score(self.y_test,logreg_tuned_pred))
        print('F1 Score: ',f1_score(self.y_test,logreg_tuned_pred))
                
    def get_svm_metrics(self):

        svm_pipeline = Pipeline(steps = [('scale',StandardScaler()),('SVM',SVC(C=1000,gamma=0.01,kernel='rbf',random_state=42))])

        svm_pipeline.fit(self.x_train_resampled,self.y_train_resampled)

        svm_tuned_pred   = svm_pipeline.predict(self.x_test)


        print(classification_report(self.y_test,svm_tuned_pred))
        print('Accuracy Score: ',accuracy_score(self.y_test,svm_tuned_pred))
        print('F1 Score: ',f1_score(self.y_test,svm_tuned_pred))
    
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
        self.logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('LR',LogisticRegression(random_state=42))])
        self.logreg_pipeline.fit(self.x_train_resampled, self.y_train_resampled)
        predictionsLR = self.logreg_pipeline.predict(self.x_test)
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

    def show_feature_importance_logistic_regression(self):
        st.subheader('Feature Importance para o Logistic Regression')
        columns_ = ['gender', 'age', 'hypertension', 'heart_disease', 'work_type',
       'avg_glucose_level', 'bmi']

        data = eli5.show_weights(self.logreg_pipeline.named_steps["LR"], feature_names=columns_).data
        data = data.replace('filter: brightness(85%);', 'filter: brightness(85%); background-color: white;')
        print(data)
        st.components.v1.html(data, height = 500)

    def hypothesis_tests_for_best_algorithm(self):
        st.subheader('Teste de hipótese para os algoritmos Random Forest e Logistic Regression')
        description_text = "Vamos analisar se o algoritmo Logistic Regression (LR) realmente possui uma diferença na métrica de performance (recall), em comparação com o Random Forest (RF) para o nosso caso. Para isso vamos efetuar um teste de hipóteses comparando ambos os modelos. Vamos ainda definir o nível de significância em 5% (0.05). Para que rejeitemos a hipótese nula, o p_value dado pela função deve ser menor ou igual ao nível de significância. Caso contrário, nós falhamos em rejeitar a hipótese nula."
        st.text_area(label='', value=description_text, height=200)
        
        # Calculates recall for Random Forest and LogisticRegression
        cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        random_forest_scores = cross_val_score(self.rf_pipeline, self.x_train_resampled, self.y_train_resampled, scoring='recall', cv=cv1, n_jobs=1)
        rf_mean = mean(random_forest_scores)
        rf_standard_deviation = std(random_forest_scores)
        st.write('Recall médio do Random Forest: %.3f (desvio padrão: %.3f)' %(rf_mean, rf_standard_deviation))
        
        logistic_regression_scores = cross_val_score(self.logreg_pipeline, self.x_train_resampled, self.y_train_resampled, scoring='recall', cv=cv1, n_jobs=1)
        lr_mean = mean(logistic_regression_scores)
        lr_standard_deviation = std(logistic_regression_scores)
        st.write('Recall médio do Logistic Regression: %.3f (desvio padrão: %.3f)' %(lr_mean, lr_standard_deviation))

        # check if difference between algorithms is real
        st.subheader('Efetuando o teste')
        t_statistics, p_value = paired_ttest_5x2cv(estimator1=self.rf_pipeline, estimator2=self.logreg_pipeline, X=self.x_train_resampled, y=self.y_train_resampled, scoring='accuracy', random_seed=1)
        alpha = 0.05
        h0 = 'Os algoritmos LR e RF provavelmente tem a mesma performance.'
        h1 = 'Os algoritmos LR e RF provavelmente não possuem a mesma performance.'
        st.write("Nível de significância (alpha): %.2f" %(alpha))
        st.write("Hipótese nula (H0): %s" %(h0))
        st.write("Hipótese alternativa (H1): %s" %(h1))
        st.write("p_value = %.5f" %(p_value))
        if p_value <= alpha:
            st.write('Como o p_value foi menor ou igual ao nível de significância, então a hipótese nula foi rejeitada. %s Logo, um algoritmo é mais relevante que o outro.' %(h1))
        else:
            st.write('O p_value foi maior que o nível de significância, logo, falhamos em rejeitar a hipótese nula. Então, %s' %(h0))

algorithms_page = Algorithms()
algorithms_page.plot_imbalanced_distribution_chart()
algorithms_page.plot_balanced_distribution_chart()
algorithms_page.calculate_score_random_forest()
algorithms_page.calculate_score_logistic_regression()
algorithms_page.calculate_score_svm()
algorithms_page.show_feature_importance_logistic_regression()
algorithms_page.hypothesis_tests_for_best_algorithm()
algorithms_page.get_rf_metrics()
algorithms_page.get_lr_metrics()
algorithms_page.get_svm_metrics()

