# -*- coding: utf-8 -*-
"""
Created on Wed Jan  31 10:31:44 2024

@author: vaish
"""

#Breast Cancer Prediction

#Load Libraries 
import numpy as np
import pandas as pd 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data collection and Processing
breast_cancer_dataset=sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)

#loading into panadas
df=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)
#first 5 columns printed
df.head(5)

#adding target column 
df['label']=breast_cancer_dataset.target
#last 5 columns printed
df.tail(5)

df.shape
#defines the dataset

df.info()
#gives all the information about dataset

df.isnull().sum()
#checking missing values 
#no missing values

df.describe()
#statistical measures about data

df['label'].value_counts()
#checking the distribution of data
#1=357=Benign cases
#0=212=malignant cases

df.groupby('label').mean()
#calculating mean

X=df.drop(columns='label',axis=1)
Y=df['label']
#separating feature and column
print(X)
print(Y)


#splitting data into training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape,X_test.shape)


model=LogisticRegression()
#training the lodistic regression with trainning data
model.fit(X_train,Y_train)

#accuracy on test data 

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy',test_data_accuracy)

#building a predective data
input_data=(20.29,14.34,135.1,1297,0.1003,0.1328,0.198,0.1043,0.1809,0.05883,0.7572,0.7813,5.438,94.44,0.01149,0.02461,0.05688,0.01885,0.01756,0.005115,22.54,16.67,152.2,1575,0.1374,0.205,0.4,0.1625,0.2364,0.07678)
#to change input data tuple to numpy array 
input_data_as_nr=np.array(input_data)
#reshaping numpy array for one datapoint 
input_data_reshaped=input_data_as_nr.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)
#so,Yes prediction is Correct 
#it has given value 0

if(prediction[0]==0):
    print('The breast cancer is Malignant')
else:
    print('The breast cancer is Benign')

    

