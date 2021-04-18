import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV,train_test_split 
from sklearn.metrics import mean_absolute_error

admissions = pd.read_csv("./dataset/Admission_Predict.csv")
admissions = admissions.drop('Serial No.',axis = 1)

list_of_uni = pd.read_csv('./dataset/university.csv')
list_of_uni = list_of_uni['school_name']

list_of_uni = list_of_uni.sample(frac=1).reset_index(drop=True)

admissions['school_name'] = list_of_uni.head(400)

print(admissions.head())


# print(admissions.describe())

# sns.pairplot(admissions)

X = admissions.drop('Chance of Admit ', axis=1)
X = X.drop('school_name', axis = 1)

print(X.head())

y = admissions['Chance of Admit ']

# print(admissions.head())

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=.25, random_state=123)

print(X_train.head())
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print('Mean absolute error for RF model: %0.4f' %
      mean_absolute_error(y_val, rf_model.predict(X_val)))

data1 = {'GRE Score': ['30'], 'TOEFL Score': ['104'], 'University Rating': ['1'], 'SOP': ['3'], 'LOR': ['3'], 'CGPA': ['8.5'], 'Research': ['0']}

df = pd.DataFrame(data1)

chances_of_admit = rf_model.predict(df)

print(chances_of_admit)


Index_label = admissions[admissions['Chance of Admit ']
                         > float(chances_of_admit)].index.tolist()



print(Index_label)


feature_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns=['Value', 'Feature']) 
plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(
    by="Value", ascending=False))
plt.show()
