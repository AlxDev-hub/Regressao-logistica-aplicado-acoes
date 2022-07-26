#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 10:38:41 2022

@author: alexsander
"""

import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

weg = yf.download('WEGE3.SA', start='2020-01-01', end='2021-10-01')

weg['Adj Close'].plot();

X = weg['Adj Close']

X_train = np.array(X).reshape(-1, 1)

media_X = np.array(X).reshape(-1, 1).mean()

y_train = []

for i in range(434):
    y = 1 if X_train[i] > media_X else 0
    y_train.append(y)

clf = LogisticRegression(solver='lbfgs').fit(X_train, y_train)

print("\nPREDIÇÃO DOS RÓTULOS PARA OS DADOS DE ENTRADA X_train:")
print(clf.predict(X_train))

print("\nESTIMATIVAS DE PROBABILIDADE:")
print(clf.predict_proba(X_train))

print("\nPRECISÃO DOS DADOS/RÓTULOS FORNECIDOS PELO DATASET:")
print(clf.score(X_train, y_train))

print("\nMatriz de confusão:")
print(confusion_matrix(y_train, clf.predict(X_train)))