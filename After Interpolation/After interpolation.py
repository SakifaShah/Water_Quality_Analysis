# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 09:44:40 2022

@author: Eutech
"""

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_excel('After interpolation.xls', index_col=0)
#print(data.head(5))
#print(data.shape)

cols = [col for col in data.columns]
cols.pop()
print(cols)
#print(cols.remove(cols[1]))
y = data.DO__mg_L_


#scikit learn
#create x and y

def linearRegression(start, end):
    #feature_cols = cols[start:end][::-1]
    feature_cols = cols[start:end]
    X = data[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    l_reg = LinearRegression()
    l_reg.fit(X_train, y_train)
    y_pred = l_reg.predict(X_test)
    r_sqr = metrics.r2_score(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    a = ''
    for i in feature_cols:
        a += i + ' '
    return [a, r_sqr, rmse]


c = 4
flag = c+1
i = c
j = len(cols)
while i < j:
    i += 1
    lr = linearRegression(c, i)
    lr_df = pd.DataFrame([lr])
    #lr_df.columns = ['Var']
    lr_df = lr_df.to_csv("lr_afterinterpolation.csv", index=False, header=False, mode='a+')
    
    if(i==j):
        cols.remove(cols[flag])
        print(cols)
        i = c+1
        j = len(cols)
 