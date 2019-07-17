# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:19:12 2019

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pandas.plotting

#Load File. Change here your file name
df=pd.read_csv('Breast_cancer_data.csv')

#create a normilised 0-1 DF
#MAKE SURE ALL COLUMNS ARE ALREADY IN NUMBERS

from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler()
n_cols=len(df.columns)-1

df_norm=pd.DataFrame(scaler.fit_transform(df.iloc[:,:n_cols]),
                          columns= df.columns.values.tolist()[:n_cols])

df_norm['target']=df.iloc[:,-1]

#Choose to use normalised or non normalised here
#Create Train and Test datasets. Make Sure Target is in the last column!
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(
        df_norm.iloc[:,:n_cols], df_norm.iloc[:,-1], stratify=df_norm.iloc[:,-1], random_state=42)


#Import and create all models. Tune appropiate parameters for each model
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4,random_state=0)

from sklearn.linear_model import LogisticRegression 
logreg = LogisticRegression(C=10)

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=1)

from sklearn.naive_bayes import BernoulliNB
naive_b = BernoulliNB(alpha=1)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=42)

from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=0.1)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='adam', random_state=42)

from sklearn.svm import SVC
svc = SVC(C=100, probability=True)


model_list=[tree,logreg,knn,naive_b,forest,gbrt,mlp,svc]

#Train Models and get score
train_scores=[]
test_scores=[]
for model in model_list:
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

scores_df= pd.DataFrame(
    {'Model Name': ['tree','logreg','knn','naive_b','forest','gbrt','mlp','svc'],
     'Train Score': train_scores,
     'Test Score': test_scores
    })

scores_df.sort_values(by=['Test Score'],ascending=False,inplace=True)    
print(scores_df)