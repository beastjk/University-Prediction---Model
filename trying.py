from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error

import pickle
import joblib

admissions = pd.read_csv("./dataset/Admission_Predict.csv")
admissions = admissions.drop('Serial No.', axis=1)
admissions = admissions.drop('Chance of Admit ', axis=1)

print(admissions.head())

list_of_uni = pd.read_csv('./dataset/university.csv')
list_of_uni = list_of_uni['school_name']

list_of_uni = list_of_uni.sample(frac=1).reset_index(drop=True)

X = admissions.drop('University_Rating', axis=1)
print(X.head())
y = admissions['University_Rating']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=.25, random_state=123)
print(X_train.head())
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model = LinearRegression()
# rf_model = Ridge(alpha=1)
rf_model = svm.SVC(kernel='linear')
rf_model.fit(X_train, y_train)



print('Mean absolute error for RF model: %0.4f' %
      mean_absolute_error(y_val, rf_model.predict(X_val)))

print('Mean squared error for RF model: %0.4f' %
      mean_squared_error(y_val, rf_model.predict(X_val)))

print(confusion_matrix(y_val, rf_model.predict(X_val)))
print(classification_report(y_val, rf_model.predict(X_val)))

filename = 'load_model.pkl'

joblib.dump(rf_model, filename)

data1 = {'GRE_Score': ['320'], 'TOEFL_Score': ['113'], 'SOP': ['4.5'], 'LOR': ['4.5'], 'CGPA': ['8.5'], 'Research': ['0']}

df = pd.DataFrame(data1)

predicted_ratings = rf_model.predict(df)

print(predicted_ratings)


Index_label = list_of_uni[admissions[admissions['University_Rating']
                         <= float(predicted_ratings)].index.tolist()]

print(Index_label)





