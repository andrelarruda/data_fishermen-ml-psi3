import pandas as pd
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
import streamlit as st

from sklearn.metrics import plot_confusion_matrix, classification_report,confusion_matrix



data = Data().get_data()


df = pd.DataFrame(data)

colunas = [coluna for coluna in df.columns if coluna not in ['id','stroke']]

conts = ['age','avg_glucose_level','bmi']

#Alterando as variáveis do tipo str para numéricos
df['gender'] = df['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
df['residence_type'] = df['residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)
df['work_type'] = df['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)
df['smoking_status'] = df['smoking_status'].replace({'smokes':1,'never smoked':0,'formerly smoked':2 ,'Unknown':-1}).astype(np.uint8)
df['ever_married'] = df['ever_married'].replace({'Yes':1,'No':0}).astype(np.uint8)


#definindo todas as variavéis 
X  = df[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']]
y = df['stroke']

#Treinando o modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

#Balanceando os dados com Smote 
oversample = SMOTE()
X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train.ravel())



#criando um novo dataframe com os dados balanceados
df_balanceado = pd.DataFrame(y_train_resh,columns=['stroke'])


#montando o daframe para exibição
depois_balanceamento=df_balanceado.groupby("stroke")["stroke"].count().reset_index(name='qtd_stroke')
antes_balanceamento=df.groupby("stroke")["stroke"].count().reset_index(name='qtd_stroke')
antes_balanceamento['stroke'] = antes_balanceamento['stroke'].replace({1:'Sim',0:'Não'}).astype(str)
depois_balanceamento['stroke'] = antes_balanceamento['stroke'].replace({1:'Sim',0:'Não'}).astype(str)
st.text_area(label='Dados não balanceados.', value='Observando o gráfico abaixo podemos notar que o número de casos confirmados de AVC é muito menor do que os casos não confirmados, dessa maneira seria difícil encontrar um modelo que trouxesse um resultado satisfatório.', height=100)
fig1 = px.bar(antes_balanceamento, x="stroke", y="qtd_stroke", color="stroke", title="Antes do balanceamento", labels={ 'qtd_stroke': 'Quantidade de ocorrências', 'stroke': 'Ocorrência de AVC', 'no_stroke': 'Não', 'yes_stroke': 'Sim' })
st.plotly_chart(fig1,use_container_width=True)

st.text_area(label='Dados  balanceados.', value='Foi utilizado o SMOTE(Synthetic Minority Oversampling Technique) para balancear os dados e assim obter melhores resultados. Como o número de casos positivos é muito menor  do que os de casos negativos, o SMOTE foi ideal para esse balanceamento, já que ele ira preencher com mais casos positivos  nosso dataframe', height=150)

fig2 = px.bar(depois_balanceamento, x="stroke", y="qtd_stroke", color="stroke", title="Depois do balanceamento", labels={ 'qtd_stroke': 'Quantidade de ocorrências', 'stroke': 'Ocorrência de AVC', 'no_stroke': 'Não', 'yes_stroke': 'Sim' })
st.plotly_chart(fig2,use_container_width=True)


# Models




rf_pipeline = Pipeline(steps = [('scale',StandardScaler()),('RF',RandomForestClassifier(random_state=42))])
svm_pipeline = Pipeline(steps = [('scale',StandardScaler()),('SVM',SVC(random_state=42))])
logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('LR',LogisticRegression(random_state=42))])

#RandomFlorest -  Treinamento

rf_pipeline.fit(X_train_resh,y_train_resh)
predictionsRF=rf_pipeline.predict(X_test)
rfcm=confusion_matrix(y_test,predictionsRF)


st.text_area(label='Matrix de Confusão Random Forest', value="",  height=1)

fig3 = px.imshow(rfcm,text_auto=True, aspect="auto",color_continuous_scale='ylgnbu',
                labels=dict(x="Valores previstos ", y="Valores reais", color="Numero de casos"),
                x=['Predição negativa', 'Predição positiva'],
                y=['negativo', 'positivo']
               )



fig3.update_xaxes(side="bottom")
st.plotly_chart(fig3)



#LogisticRegression -  Treinamento

logreg_pipeline.fit(X_train_resh,y_train_resh)
predictionsLR=logreg_pipeline.predict(X_test)
lgrmc= confusion_matrix(y_test,predictionsLR)

st.text_area(label='Matrix de Confusão Logistic Regression',  value="" ,  height=1)

fig4 = px.imshow(lgrmc,text_auto=True, aspect="auto",color_continuous_scale='ylgnbu',
                labels=dict(x="Valores previstos ", y="Valores reais", color="Numero de casos"),
                x=['Predição negativa', 'Predição positiva'],
                y=['negativo', 'positivo']
               )



fig4.update_xaxes(side="bottom")
st.plotly_chart(fig4)




#SVC - Treinamento

svm_pipeline.fit(X_train_resh,y_train_resh)
predictionsSVC=svm_pipeline.predict(X_test)
svmcm=confusion_matrix(y_test,predictionsSVC)

#matriz de confusão
st.text_area(label='Matrix de Confusão SVC',  value="" , height=1)

fig5 = px.imshow(svmcm,text_auto=True, aspect="auto",color_continuous_scale='ylgnbu',
                labels=dict(x="Valores previstos ", y="Valores reais", color="Numero de casos"),
                x=['Predição negativa', 'Predição positiva'],
                y=['negativo', 'positivo']
               )



fig5.update_xaxes(side="bottom")
st.plotly_chart(fig5)










#o Cross_val_score serve para ver a relação entre a precisão e o recall
rf_cv = cross_val_score(rf_pipeline,X_train_resh,y_train_resh,cv=10,scoring='f1')
svm_cv = cross_val_score(svm_pipeline,X_train_resh,y_train_resh,cv=10,scoring='f1')
logreg_cv = cross_val_score(logreg_pipeline,X_train_resh,y_train_resh,cv=10,scoring='f1')

print('Mean f1 scores:')
print('Random Forest mean :',rf_cv.mean())
print('SVM mean :',svm_cv.mean())
print('Logistic Regression mean :',logreg_cv.mean())

#classificador
colors = ["lightgray","lightgray","#0f4c81"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
def rf_feat_importance(m, df):
    return pd.DataFrame({'Feature':df.columns, 'Importance':m.feature_importances_}).sort_values('Importance', ascending=False)

#Obtendo a Feature importance do Random Forest
fi_random_florest = rf_feat_importance(rf_pipeline['RF'], X)


fig6 = px.bar(fi_random_florest, y='Importance', x='Feature',
            title="Feature Importance")
st.plotly_chart(fig6)







