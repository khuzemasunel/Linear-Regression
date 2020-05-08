# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:22:54 2020

@author: Khuzema sunel
Last edited : 4:20pm 5/18/2020
"""

from GetData import GetData
from Utility import Util
from Transform import Transform
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


#read config
config = Util.read_config()
#get data
df = GetData.read_csv(GetData,config['data']['FuelConsumption']['filepath'])
#get all numeric columns
cols_numeric = df.select_dtypes([np.number]).columns
#remove non-numeric columns
df = df[cols_numeric]
#find correlations between variables
corrMatrix = df.corr()
mask = np.zeros_like(corrMatrix)
mask[np.triu_indices_from(mask)] = True
with sn.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    sn.heatmap(corrMatrix,vmin=-1,vmax=1,mask=mask,annot=True,linewidths=.5,cmap="Greens")

plt.scatter(df.FUELCONSUMPTION_CITY,df.CO2EMISSIONS,color = 'green')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

df = df.drop('MODELYEAR',axis=1)
#scatter plots independant variables which have high corr with CO2Emission (dependant variable)
groups = df.groupby('CYLINDERS')
for name, group in groups:
    plt.plot(group['ENGINESIZE'],group['CO2EMISSIONS'],marker ="o",linestyle="", label=name)
plt.xlabel("ENGINE SIZE")
plt.ylabel("CO2 EMISSION")
plt.legend(title='CYLINDERS')
plt.show()
for name, group in groups:
    plt.plot(group['FUELCONSUMPTION_COMB_MPG'],group['CO2EMISSIONS'],marker ="o",linestyle="", label=name)
plt.xlabel("FUEL CONSUMPTION COMBINED")
plt.ylabel("CO2 EMISSION")
plt.legend(title='CYLINDERS')
plt.show()

#create test and train datasets
ds = Transform.createTrainTest(Transform,df,0.8)
X_train = ds['X_train'][['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
X_test = ds['X_test'][['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
Y_train = ds['Y_train'][['CO2EMISSIONS']]
Y_test = ds['Y_test'][['CO2EMISSIONS']]

#Identify Top performing models
simple = linear_model.LinearRegression()
ridge = linear_model.Ridge()
lasso = linear_model.Lasso()
elastic = linear_model.ElasticNet()
lasso_lars = linear_model.LassoLars()
bayesian_ridge = linear_model.BayesianRidge()

models = [ridge, lasso, elastic, lasso_lars, bayesian_ridge]

def get_cv_scores(model):
    scores = cross_val_score(model, X_train, Y_train)
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')

for model in models:
    print(model)
    get_cv_scores(model)

#multiple regression model
regr = linear_model.LinearRegression()
x = np.asanyarray(X_train)
y = np.asanyarray(Y_train)
ridge.fit (x, y)
# The coefficients
print ('Coefficients: ', ridge.coef_)
print ('Intercept: ', ridge.intercept_)

#prediction
y_hat= ridge.predict(X_test)
x = np.asanyarray(X_test)
y = np.asanyarray(Y_test)
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % ridge.score(x, y))


