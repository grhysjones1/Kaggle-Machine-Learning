#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:03:11 2018

@author: garethjones
"""

''' IMPORTS & FILE READ '''
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


directory = '/Users/garethjones/Documents/Data Analysis/Kaggle Intro/Data/'
file = 'train.csv' 
data = pd.read_csv(directory+file)


''' CLEAN DATA '''
# A nice way to write a for loop and if statement
cols_with_missing = [col for col in data.columns if data[col].isnull().any()]

# Use Imputer function to fill in NAN values with mean for that column
my_imputer = Imputer()
data_imputed = my_imputer.fit_transform(data)


''' SETUP TEST AND TRAIN VARIABLES '''
# These are the variables we will use to predict something else
predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = data[predictors]

# This is what we want to predict
y = data.SalePrice

# Split our dataset into training and testing data
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size = 0.7, test_size = 0.3, random_state = 0)


''' SIMPLE DECISION TREE MODEL '''
# Choose a model type and fit it using training data
dt_model = DecisionTreeRegressor()
dt_model.fit(train_X,train_y)

# Predict home prices using testing data
dt_pred_prices = dt_model.predict(val_X)

# Work out what mean error of the model is
dt_mae = mean_absolute_error(val_y, dt_pred_prices)
dt_mae = '{0:.0f}'.format(dt_mae)
print('The mean absolute error of the simple decision tree model is $'+dt_mae)


''' DECISION TREE WITH MAX LEAF NODES '''
# Create a function that returns the MAE for a given number of leaf nodes
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# Write a for loop to input a number of different leaf node values, and see what the lowest MAE is
dt_maelist=[]
for max_leaf_nodes in [5,50,500,5000]:
    mae = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    mae = '{0:.0f}'.format(mae)
    dt_maelist.append(mae)
print(dt_maelist)


''' RANDOM FOREST MODEL '''
rf_model = RandomForestRegressor()
rf_model.fit(train_X,train_y)
rf_pred_prices = rf_model.predict(val_X)
rf_mae = mean_absolute_error(val_y,rf_pred_prices)
rf_mae = '{0:.0f}'.format(rf_mae)
print('The mean absolute error of the random forest model is $'+rf_mae)



