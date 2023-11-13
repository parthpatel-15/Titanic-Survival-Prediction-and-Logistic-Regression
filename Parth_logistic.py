#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 02:39:39 2021

@author: Parth Patel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# load data
filename='titanic.csv'
path='/Users/sambhu/iCloud Drive (Archive) - 1/Desktop/centennial college /sem-1/intoduction to AI'
fullpath=os.path.join(path,filename)
titaninc_parth  = pd.read_csv(fullpath)

# Data Analysis 
    
print(titaninc_parth.head(3))
print(titaninc_parth.shape)
print(titaninc_parth.info)
print(titaninc_parth.isna().sum(),titaninc_parth.dtypes)
print (titaninc_parth.isnull().sum())
print(titaninc_parth['Sex'].value_counts())
print(titaninc_parth['Pclass'].value_counts())

# data Visulization 

pd.crosstab(titaninc_parth.Survived,titaninc_parth.Pclass).plot(kind='bar')
plt.title('Survival in different classes')
plt.xlabel('Passenger class for (not survived=0) and (survived=1)')
plt.ylabel('Number of people ')

pd.crosstab(titaninc_parth.Survived,titaninc_parth.Sex).plot(kind='bar')
plt.title('Survival of different gender')
plt.xlabel('Gender of (not survived=0) and (survived=1)')
plt.ylabel('Number of people ')

cat_vars=['Sex','Pclass','Fare','SibSp','Parch']
titaninc_parth_vars=titaninc_parth.columns.values.tolist()
to_keep=[i for i in titaninc_parth_vars if i in cat_vars]
titaninc_parth_final=titaninc_parth[to_keep]
titaninc_parth_final.columns.values

pd.plotting.scatter_matrix(titaninc_parth_final)

# Data Cleaning 
    
titaninc_parth_f=titaninc_parth.drop(columns=['PassengerId','Name', 'Ticket', 'Cabin'])
print(titaninc_parth_f.shape)
# make dummies 
col1=['Sex','Embarked']
for var in col1:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(titaninc_parth_f[var], prefix=var)
# 	Attach the newly created variables 
    titaninc_parth_f1=titaninc_parth_f.join(cat_list)
    titaninc_parth_f=titaninc_parth_f1
    
    
    #drop column 
titaninc_parth_f=titaninc_parth_f1.drop(columns=['Sex','Embarked'])
print(titaninc_parth_f.shape)


# add age at null data
age_mean=titaninc_parth['Age'].mean()
titaninc_parth_f["Age"].fillna(age_mean, inplace = True)

print (titaninc_parth_f.isnull().sum())


# convert all data in float
titaninc_parth_f = titaninc_parth_f.astype(float)
print(titaninc_parth_f.dtypes)

titaninc_parth_f.info()

# normalization func
def normalization(df):
       df = df.copy()
   
       for column in df.columns:
           df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        
       return df


    
titaninc_parth_norm=normalization(titaninc_parth_f)

print(titaninc_parth_norm.head(2))

titaninc_parth_norm.hist(figsize=(9,10))


# Logistic regreation 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score

#make featues column and target
feature_cols1 = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X_parth = titaninc_parth_norm[feature_cols1]
Y_parth = titaninc_parth_norm['Survived']

print(X_parth.shape)

#slpit data
x_train_Parth,x_test_Parth,y_train_Parth,y_test_Parth = train_test_split(X_parth,Y_parth, test_size = 0.30)

#create model 
np.random.seed(43)
parth_model = linear_model.LogisticRegression(solver='lbfgs')
parth_model.fit(x_train_Parth,y_train_Parth)
print (parth_model.coef_)
print (parth_model.score(x_train_Parth,y_train_Parth))

for i in np.arange (0.10, 0.55, 0.05): 
    print("i:",i)
    x_train_Parth,x_test_Parth,y_train_Parth,y_test_Parth = train_test_split(X_parth,Y_parth, test_size = i)

 
    scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), x_train_Parth, y_train_Parth, scoring='accuracy', cv=10)
    print ("Minimum : ",scores.min())
    print ("Mean : ",scores.mean())
    print ("Maximum : ",scores.max())
    print ("--------------*-----------*----------------")


# Test the Model
x_train_Parth,x_test_Parth,y_train_Parth,y_test_Parth = train_test_split(X_parth,Y_parth, test_size = 0.30)
np.random.seed(43)
parth_model = linear_model.LogisticRegression(solver='lbfgs')
parth_model.fit(x_train_Parth,y_train_Parth)
y_pred_parth=parth_model.predict_proba(x_test_Parth)
print(y_pred_parth)

y_pred_parth_flag = y_pred_parth[:,1] > 0.5
print(y_pred_parth_flag)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


#4-Check model accuracy, create consion matrix, classification report
print (accuracy_score(y_test_Parth, y_pred_parth_flag))


print (confusion_matrix(y_test_Parth, y_pred_parth_flag))


print(classification_report(y_test_Parth, y_pred_parth_flag))


#set threshold to 0.75
y_pred_parth_flag = y_pred_parth[:,1] > 0.75
print(y_pred_parth_flag)


#4-Check model accuracy, create consion matrix, classification report
print (accuracy_score(y_test_Parth, y_pred_parth_flag))


print (confusion_matrix(y_test_Parth, y_pred_parth_flag))


print(classification_report(y_test_Parth, y_pred_parth_flag))



