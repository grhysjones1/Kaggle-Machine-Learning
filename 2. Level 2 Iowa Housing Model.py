#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:35:49 2018

@author: garethjones
"""


''' 0. IMPORTS & FILE READ '''
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score



''' 1. PREPARE DATA '''
directory = '/Users/garethjones/Documents/Data Analysis/Kaggle Intro/Data/'
file = 'train.csv' 
data = pd.read_csv(directory+file)

''' Split Dataset '''
data_cols_numeric = data.select_dtypes(exclude='object').columns.sort_values().tolist()
data_cols_categorical = data.select_dtypes('object').columns.sort_values().tolist()
# Target
y = data.SalePrice
# Predictors (all numeric and then some categoricals)
X = data[data_cols_numeric].drop(['SalePrice'],axis=1)
X['HeatingQC'] = data[['HeatingQC']]
# Split data into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)

''' Create One Hot Encoded Dataset '''
# You cannot impute categorical data, so we need to do one_hot_encoding
X_one_hot = pd.get_dummies(X)
X_train_one_hot = pd.get_dummies(X_train)
X_test_one_hot = pd.get_dummies(X_test)

# Ensure same number and ordering of columns in the one_hot_encoded test and train datasets
X_train_one_hot, X_test_one_hot = X_train_one_hot.align(X_test_one_hot,join='left',axis=1)
X_test_one_hot['HeatingQC_Po'] = X_test_one_hot['HeatingQC_Po'].fillna(value=0) # Try and automate finding the column that has zeros in it


''' 2. DEFINE FUNCTIONS '''

''' 2a. Score Random Forest, Simple MAE (Drop Nan Columns) '''
def get_mae_dropnans(X_train,X_test,y_train,y_test):
    model = RandomForestRegressor()
    cols_with_nans = [col for col in X.columns if X[col].isnull().any()]
    X_train_reduced = X_train.drop(cols_with_nans,axis=1)
    X_test_reduced = X_test.drop(cols_with_nans,axis=1)
    model.fit(X_train_reduced,y_train)
    predict_prices = model.predict(X_test_reduced)
    mae = mean_absolute_error(y_test,predict_prices)
    return(mae)
    
''' 2b. Score Random Forest, Simple MAE (Impute Nan Columns) '''
def get_mae_imputednans(X_train,X_test,y_train,y_test):
    model = RandomForestRegressor()
    my_imputer = Imputer()
    X_train_imputed = my_imputer.fit_transform(X_train)
    X_test_imputed = my_imputer.fit_transform(X_test)
    model.fit(X_train_imputed,y_train)
    predict_prices = model.predict(X_test_imputed)
    mae = mean_absolute_error(y_test,predict_prices)
    return(mae)
    
''' 2c. Cross Validate Random Forest (Drop Nan Columns) '''
def cv_get_mae_dropnans(X,y):
    model = RandomForestRegressor()
    cols_with_nans = [col for col in X.columns if X[col].isnull().any()]
    X_dropped = X.drop(cols_with_nans,axis=1)
    mae_avg = -1*cross_val_score(model,X_dropped,y,scoring='neg_mean_absolute_error').mean()
    return(mae_avg)
    
''' 2d. Cross Validate Random Forest (Impute Nan Columns) '''
def cv_get_mae_imputednans(X,y):
    model = RandomForestRegressor()
    my_imputer = Imputer()
    X_imputed = my_imputer.fit_transform(X)
    mae_avg = -1*cross_val_score(model,X_imputed,y,scoring='neg_mean_absolute_error').mean()
    return(mae_avg)
    

    
''' 3. SCORE MODELS BY MAE '''
# get_mae without cv requires input of split datasets
print('Mean Absolute Error from dropping Nan columns:')
print(int(get_mae_dropnans(X_train_one_hot,X_test_one_hot,y_train,y_test)))

print('Mean Absolute Error from imputing Nan columns:')
print(int(get_mae_imputednans(X_train_one_hot,X_test_one_hot,y_train,y_test)))



''' 4. SCORE MODELS BY CROSS VALIDATION '''
# cv_get_mae formulas don't need split datasets, as this is included in the cross_val_score function
print('Mean Cross Validated Score from dropping Nan columns:')
print(int(cv_get_mae_dropnans(X_one_hot,y)))

print('Mean Cross Validated Score from imputing Nan columns:')
print(int(cv_get_mae_imputednans(X_one_hot,y)))
