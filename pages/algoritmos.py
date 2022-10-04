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
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression,LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report,confusion_matrix,f1_score, classification_report
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score, roc_curve, auc
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
        self.rf_pipeline = Pipeline(steps = [('scale',StandardScaler()),('RF',RandomForestClassifier(random_state=42, n_estimators=750, min_samples_split=3, min_samples_leaf=3, max_features='sqrt', max_depth=2, bootstrap=True))])

        self.rf_pipeline.fit(self.x_train_resampled, self.y_train_resampled)
        predictionsRF = self.rf_pipeline.predict(self.x_test)
        rfcm = confusion_matrix(self.y_test, predictionsRF)
        dict_rf = classification_report(self.y_test,predictionsRF,output_dict=True)
        self.matrix(rfcm, 'Random Forest', dict_rf)

    def calculate_score_logistic_regression(self):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        self.logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('LR',LogisticRegression(solver='newton-cg', penalty='l2', C=0.0000000001))])
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

    def calculate_score_xgboost(self):
        xgboost_pipeline = Pipeline(steps = [('scale',StandardScaler()),('XGBoost',XGBClassifier(random_state=23, max_depth=5, learning_rate=0.0005, n_estimators=170, colsample_bytree=0.007))])
        xgboost_pipeline.fit(self.x_train_resampled, self.y_train_resampled)

        xgboost_predictions = xgboost_pipeline.predict(self.x_test)
        xgb_cm = confusion_matrix(self.y_test, xgboost_predictions)
        dict_xgb = classification_report(self.y_test, xgboost_predictions, output_dict=True)

        self.matrix(xgb_cm, 'XGBoost', dict_xgb)

    def tune_hyperparameters(self, algorithm, cv):
        if algorithm == 'rf':
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            random_state = [x for x in range(1, 44)]
            params = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap,
                    'random_state': random_state,
            }
            rf_random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=params, scoring='recall', cv=3, n_iter=200, verbose=1, random_state=42, n_jobs=-1)
            rf_random_search.fit(self.x_train_resampled, self.y_train_resampled)

            st.write("Best parameters:", rf_random_search.best_params_)
            st.write("Lowest RMSE: ", (-rf_random_search.best_score_)**(1/2.0))
        elif algorithm == 'lr':
            model = LogisticRegression()
            solvers = ['newton-cg', 'lbfgs', 'liblinear']
            penalty = ['l2']
            c_values = [100, 10, 1.0, 0.1, 0.01]
            # define grid search
            rs_dict = dict(solver=solvers,penalty=penalty,C=c_values)
            lr_random_search = RandomizedSearchCV(estimator=model, param_distributions=rs_dict, n_jobs=-1, cv=cv, scoring='recall', error_score=0, n_iter=100)
            lr_random_result = lr_random_search.fit(self.x_train_resampled, self.y_train_resampled)

            # summarize results
            print("Best: %f using %s" % (lr_random_result.best_score_, lr_random_result.best_params_))
            means = lr_random_result.cv_results_['mean_test_score']
            stds = lr_random_result.cv_results_['std_test_score']
            params = lr_random_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

        elif algorithm == 'svm':
            svm_random_search = RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=params, scoring='recall', n_iter=60, verbose=1)
            svm_random_search.fit(self.x_train_resampled, self.y_train_resampled)

            st.write("Best parameters:", svm_random_search.best_params_)
            st.write("Lowest RMSE: ", (-svm_random_search.best_score_)**(1/2.0))
        elif algorithm == 'xgb':
            randomSearch = RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=params, scoring='recall', n_iter=60, verbose=1)
            randomSearch.fit(self.x_train_resampled, self.y_train_resampled)

            st.write("Best parameters:", randomSearch.best_params_)
            st.write("Lowest RMSE: ", (-randomSearch.best_score_)**(1/2.0))

    def tune_hyperparameters_using_grid_search(self, algorithm, cv):
        params = {
            'random_state' : [10, 23, 38, 42, 50],
            'max_depth' : [7, 8, 10, 12, 15, 20],
            'learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9],
            'n_estimators' : [50, 150, 300, 400, 600, 700],
            'colsample_bytree' : [0.08, 0.009, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 4, 8, 10, 12],
            # 'min_samples_leaf' : [1, 2, 3],
        }

        if algorithm == 'rf':
            model = RandomForestClassifier()
        elif algorithm == 'lr':
            model = LogisticRegression()
            solvers = ['newton-cg', 'lbfgs', 'liblinear']
            penalty = ['l2']
            c_values = [100, 10, 1.0, 0.1, 0.01]
            # define grid search
            rs_dict = dict(solver=solvers,penalty=penalty,C=c_values)
            lr_random_search = GridSearchCV(estimator=model, param_grid=rs_dict, n_jobs=-1, cv=cv, scoring='recall', error_score=0)
            lr_random_result = lr_random_search.fit(self.x_train_resampled, self.y_train_resampled)

            # summarize results
            st.write("Best: %f using %s" % (lr_random_result.best_score_, lr_random_result.best_params_))
            means = lr_random_result.cv_results_['mean_test_score']
            stds = lr_random_result.cv_results_['std_test_score']
            params = lr_random_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                st.write("%f (%f) with: %r" % (mean, stdev, param))
        elif algorithm == 'svm':
            model = SVC()
        elif algorithm == 'xgb':
            model = XGBClassifier()
            # st.write(" Results from Grid Search " )
            # st.write("The best estimator across ALL searched params: ",grid_GBC.best_estimator_)
            # st.write("The best score across ALL searched params: ",grid_GBC.best_score_)
            # st.write("The best parameters across ALL searched params: ",grid_GBC.best_params_)

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
        description_text = "Vamos analisar se o algoritmo Logistic Regression (LR) realmente possui uma diferença na métrica de performance (recall), em comparação com o Random Forest (RF) para o nosso caso. Para isso vamos efetuar um teste de hipóteses comparando ambos os modelos. Vamos ainda definir o nível de significância em 5% (0.05). Para que rejeitemos a hipótese nula, o p-value (probability value) dado pela função deve ser menor ou igual ao nível de significância. Caso contrário, nós falhamos em rejeitar a hipótese nula."
        st.text_area(label='', value=description_text, height=200)
        
        # Calculates recall for Random Forest and LogisticRegression
        cv1 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
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
        st.write("Valor de probabilidade (p-value) = %.5f" %(p_value))
        if p_value <= alpha:
            st.write('Como o p-value foi menor ou igual ao nível de significância, então a hipótese nula foi rejeitada. %s Logo, um algoritmo é mais relevante que o outro.' %(h1))
        else:
            st.write('O p-value foi maior que o nível de significância, logo, falhamos em rejeitar a hipótese nula. Então, %s' %(h0))

algorithms_page = Algorithms()
# algorithms_page.plot_imbalanced_distribution_chart()
# algorithms_page.plot_balanced_distribution_chart()
# algorithms_page.calculate_score_random_forest()
algorithms_page.calculate_score_logistic_regression()
# algorithms_page.calculate_score_svm()
# algorithms_page.calculate_score_xgboost()
# algorithms_page.show_feature_importance_logistic_regression()
# algorithms_page.hypothesis_tests_for_best_algorithm()
# algorithms_page.get_rf_metrics()
# algorithms_page.get_lr_metrics()
# algorithms_page.get_svm_metrics()

