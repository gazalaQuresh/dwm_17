# -*- coding: utf-8 -*-

import numpy as np
#simport pandas as pd 
import matplottlib as mpl
import matplotlib.pyplotlib.pylot as plt
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1:].values
from sklearn.model_selection import 
train_test_split 
#used model_selection in place of cross_validation since the latter is deprecated 
X_train,X_test,y_train,y_test=
train_test_split(X,y,test_size=1/3,random+state = 0)
#fitting simple linear regression to training set
from sklearn.linear_model import 
LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#predicting the test set result
y_pred = regressor.predict(X_test)