#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import MissingIndicator
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression                         
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors                           
from sklearn import tree, linear_model                         
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis





df = pd.read_csv('heloc_dataset_v1.csv')
n_rows = df.shape[0]
n_cols = df.shape[1]
col_names = df.columns
row_indexes = df.index.tolist()
df_first3rows = df.head(3)
df_last3rows = df.tail(3)
df_rows1and3 = df.loc[[1, 3]]
df_cols2and4 = df.iloc[:, [2, 4]]
series_customer5 = df.loc[5]
df_first3rowsAndCols = df.iloc[:3, :3]
n_cols_with_missing_values = df.isnull().any(axis=0).sum()
n_rows_with_missing_ExternalRiskEstimate_values = df['ExternalRiskEstimate'].isnull().sum()
cols_numeric = []
cols_string = []
for col in col_names:
    if df[col].dtype in ['int64', 'float64']:
        cols_numeric.append(col)
    elif df[col].dtype == 'object':
        cols_string.append(col)
df_missing_ExternalRiskEstimate = df[df['ExternalRiskEstimate'] == -9]
n_rows_with_missing_ExternalRiskEstimate = df_missing_ExternalRiskEstimate.shape[0]
df_missing_ExternalRiskEstimate_replaced = df_missing_ExternalRiskEstimate.replace(-9, np.nan)
n_rows_all_numeric_missing = df_missing_ExternalRiskEstimate_replaced.select_dtypes(include=np.number).isnull().all(axis=1).sum()
df_replaced = df.replace(-9, np.nan)
df_without_missing_rows = df[df_replaced[cols_numeric].notna().any(axis=1)]
s_minus_7 = (df_without_missing_rows == -7).sum()
s_minus_8 = (df_without_missing_rows == -8).sum()
s_minus_9 = (df_without_missing_rows == -9).sum()
s1 = pd.Series([True,True,False,False])
s2 = pd.Series([True,False,True,False])
#s1|s2
def has_missing_value(row):
    return any(row.isin([-7, -8, -9]))
s_some_values_are_missing = df_without_missing_rows.apply(has_missing_value, axis=1)
grouped = df_without_missing_rows.groupby('RiskPerformance').mean()
grouped = grouped.T
grouped.columns = ['Bad', 'Good']
df_avg_feature_value_per_group = grouped
df_without_missing_rows.head()
df = df_without_missing_rows
first_col = df.pop('RiskPerformance')
df.insert(len(df.columns), 'RiskPerformance', first_col)
X = df.iloc[:, :23]
Y = df.iloc[:, -1]
Y = Y.replace({"Bad": 1, "Good": 0})
default_probability = Y.mean()
df_all = df
df_all.iloc[:, -1] = Y
df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=1234)
X_train = df_train.iloc[:, :23]
X_test  = df_test.iloc[:, :23]
Y_train = df_train.iloc[:, -1]
Y_test  = df_test.iloc[:, -1]
missing_train = X_train['ExternalRiskEstimate'] == -9
X_train = X_train.loc[~missing_train, :]
Y_train = Y_train.loc[~missing_train]
missing_test = X_test['ExternalRiskEstimate'] == -9
X_test = X_test.loc[~missing_test, :]
Y_test = Y_test.loc[~missing_test]
df_count_missing = pd.concat([(X_train==-7).sum(), (X_train==-8).sum(), (X_train==-9).sum()], axis=1)
df_count_missing.columns = [-7,-8,-9]
do_nothing_imputer = ColumnTransformer([("Imputer -7 to mean", SimpleImputer(missing_values=-7, strategy='mean'), [])], remainder='passthrough')
feature_expansion = FeatureUnion([("do nothing", do_nothing_imputer),
                                  ("add features for -7", MissingIndicator(missing_values=-7, features='missing-only')),
                                  ("add features for -8", MissingIndicator(missing_values=-8, features='missing-only'))])
pipeline = Pipeline([("expand features", feature_expansion), 
                 ("replace -7 with -8", SimpleImputer(missing_values=-7, strategy='constant', fill_value=-8)),
                 ("replace -8 with mean", SimpleImputer(missing_values=-8, strategy='mean'))])
arr_X_train_t = pipeline.fit_transform(X_train)
minus_7_indicator_transformer = MissingIndicator(missing_values=-7, features='missing-only').fit(X_train)
col_names_minus_7 = X_train.columns.values[minus_7_indicator_transformer.features_].tolist() 
col_names_minus_7 = list(map(lambda s:str(s)+'=-7',col_names_minus_7)) 
minus_8_indicator_transformer = MissingIndicator(missing_values=-8, features='missing-only').fit(X_train)
col_names_minus_8 = X_train.columns.values[minus_8_indicator_transformer.features_].tolist()
col_names_minus_8 = list(map(lambda s:str(s)+'=-8',col_names_minus_8))
column_names = X_train.columns.values.tolist() + col_names_minus_7 + col_names_minus_8
X_train_t = pd.DataFrame(arr_X_train_t, columns = column_names)
new_data =  X_test.iloc[[3,7],:]
new_data_t = pipeline.transform(new_data) # Notice that we run transform() and not fit_transform()!
new_data_t = pd.DataFrame(new_data_t, columns=column_names)
X_test_t = pipeline.transform(X_test) # Notice that we run transform() and not fit_transform()!
X_test_t = pd.DataFrame(X_test_t, columns=column_names)





X_train_t_tr, X_train_t_val, Y_train_t_tr, Y_train_t_val = train_test_split(X_train_t, Y_train, test_size=0.25, random_state=1234)

n = len(Y_train_t_tr) + len(Y_train_t_val) + len(Y_test)

#log_reg = LogisticRegression(max_iter=10000, random_state=0).fit(X_train_t_tr, Y_train_t_tr) # Logistic regression



#param_grid = {'C': np.arange(0.01, 1, 0.01)}
#log_reg = LogisticRegression(max_iter=10000, random_state=0)
#grid_search = GridSearchCV(log_reg, param_grid, cv=10)
#grid_search.fit(X_train_t, Y_train)
#log_reg_best = LogisticRegression(max_iter=10000, random_state=0)
#log_reg_best.set_params(C=0.09)
#for i in range(15,35):
selector = SelectKBest(chi2, k=24)
X_train_t_selected = selector.fit_transform(X_train_t, Y_train)
X_test_t_selected = selector.transform(X_test_t)
log_reg_best = LogisticRegression(C=0.09, max_iter=10000, random_state=0).fit(X_train_t_selected, Y_train)
test_accuracy = log_reg_best.score(X_test_t_selected, Y_test)



log_reg_best.fit(X_train_t_selected, Y_train)



pipeline = Pipeline([('selector', selector), ('log_reg_best', log_reg_best)])
pipeline.fit(X_train_t_selected, Y_train)
y_pred = pipeline.predict(X_test_t_selected)
probs = pipeline.predict_proba(X_train_t_selected)
test_accuracy = accuracy_score(Y_test, y_pred)



# In[2]:


with open('\\Users\\liqin\\Desktop\\predp\\team\\knn_model2.p', 'rb') as f2:
    loaded_model = pickle.load(f2)
st.header('Risk Predictor for Home Equity Line of Credit')
st.text('This interface has been designed to assist banks in evaluating a client\'s Home Equity\nLine of Credit.\nThe user simply inputs the application profile values for each feature, and based on\nthe weighting of each feature, a corresponding risk level comment is provided.\nThis allows the bank to determine the probability of how low/high-risky on the\napplication and make an informed decision about whether to proceed with the application.')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


x1 = st.slider('Consolidated Version of Risk Markers', 35,100,35,1)


x2 = st.slider('Months Since Oldest Trade Open', 5,610,5,1)
x3 = st.slider('Months Since Most Recent Trade Open', 0,185,0,1)
x4 = st.slider('Average Months in File', 0,260,0,1)
x5 = st.slider('Number Satisfactory Trades', 0,80,0,1)
x6 = st.slider('Number Trades 60+ Ever', 0,20,0,1)
x7 = st.slider('Number Trades 90+ Ever', 0,20,0,1)
x8 = st.slider('Percent Trades Never Delinquent', 0,100,0,1)
x9 = st.slider('Months Since Most Recent Delinquency', 0,85,0,1)
x10 = st.slider('Max Delq/Public Records Last 12 Months', 0,10,0,1)
x11 = st.slider('Max Delinquency Ever', 0,10,0,1)
x12 = st.slider('Total Number of Credit Accounts', 0,100,0,1)
x13 = st.slider('Number of Trades Open in Last 12 Months', 0,20,0,1)
x14 = st.slider('Percent Installment Trades', 0,100,0,1)
x15 = st.slider('Months Since Most Recent Inquiry Excluding 7days', 0,24,0,1)
x16 = st.slider('Number of Inquiry Last 6 Months', 0,30,0,1)
x17 = st.slider('Number of Inquiry Last 6 Months Excluding 7days', 0,30,0,1)
x18 = st.slider('Net Fraction Revolving Burden', 0,240,0,1)
x19 = st.slider('Net Fraction Installment Burden', 0,475,0,1)
x20 = st.slider('Number Revolving Trades with Balance', 0,24,0,1)
#x21 = st.slider('NumInstallTradesWBalance', 0,20,0,1)

x21 = st.slider('Number Bank Trades with High Utilization Ratio', 0,20,0,1)
x22 = st.slider('Percent Trades with Balance', 0,100,0,1)
x23 = st.slider('If There Is No Months Since Most Recent Delinquency', 0,1,0,1) #MSinceMostRecentDelq=-7 9


#x25 = st.slider('MSinceMostRecentInqexcl7days=-7', 0,1,0,1)
#x26 = st.slider('MSinceOldestTradeOpen=-8', 5.0,605.0,5.0,0.1)
#x27 = st.slider('MSinceMostRecentDelq=-8', 5.0,605.0,5.0,0.1)
x24 = st.slider('If There Are Non-usable/unvalid Trades or Inquiries', 0,1,0,1) #MSinceMostRecentInqexcl7days=-8 15
#x27 = st.slider('NetFractionRevolvingBurden=-8', 0,1,0,1)
#x28 = st.slider('NetFractionInstallBurden=-8', 0,1,0,1)
#x29 = st.slider('NumRevolvingTradesWBalance=-8', 0,1,0,1)
#x32 = st.slider('NumInstallTradesWBalance=-8', 5.0,605.0,5.0,0.1)
#x30 = st.slider('NumBank2NatlTradesWHighUtilization=-8', 0,1,0,1)
#x34 = st.slider('PercentTradesWBalance=-8', 5.0,605.0,5.0,0.1)



input_value = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24]
#X_values = X_train_t_selected
#print(loaded_model.predict(X_values[:3,:])) # make predictions for the first 3 observations
prediction = loaded_model.predict_proba([input_value])[0]


if prediction[0] > prediction[1]:
    st.text('This application has a %.2f%% probability to be low risk.' % (prediction[0]*100))
else:
    st.text('This application has a %.2f%% probability to be high risk.' % (prediction[1]*100))



# In[ ]:




