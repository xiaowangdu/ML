
# coding: utf-8

# In[32]:

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split

# load dataset
titanic_train = pd.read_csv('../dataset/titanic/train.csv')
titanic_test = pd.read_csv('../dataset/titanic/test.csv')

titanic_train.info()
titanic_test.info()


# In[33]:

#make train dataset

predict_col = ['Survived']
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']


X_train = titanic_train[feature_cols]
X_train['Sex'] = X_train['Sex'].apply(lambda x: int(x == 'male'))
X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)

y_train = titanic_train[predict_col]

X_test = titanic_test[feature_cols]
X_test['Sex'] = X_test['Sex'].apply(lambda x: int(x == 'male'))
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)

x_test = pd.read_csv('../dataset/titanic/gender_submission.csv')['Survived']

X_train.info()
X_test.info()


# In[75]:

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 5, min_samples_split = 5)
model.fit(X_train, y_train)
print(model)

# make predictions
expected = x_test
predicted = model.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

predicted = pd.DataFrame(predicted, columns = ['Survived'])
predicted['PassengerId'] = pd.read_csv('../dataset/titanic/gender_submission.csv')['PassengerId']

predicted = predicted.reindex(columns = ['PassengerId', 'Survived'])
predicted.to_csv("./output/submission.csv", index=False)


# In[76]:

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
print(model)

# make predictions
expected = x_test
predicted = model.predict(X_test)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

predicted = pd.DataFrame(predicted, columns = ['Survived'])
predicted['PassengerId'] = pd.read_csv('../dataset/titanic/gender_submission.csv')['PassengerId']

predicted = predicted.reindex(columns = ['PassengerId', 'Survived'])
predicted.to_csv("./output/submission.csv", index=False)


# In[ ]:

df = loan_2.reindex(columns= ['term_clean','grade_clean', 'annual_inc', 'loan_amnt', 'int_rate','purpose_clean','installment','loan_status_clean'])
df.fillna(method= 'ffill').astype(int)
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
array = df.values
y = df['loan_status_clean'].values
imp.fit(array)
array_imp = imp.transform(array)

y2= y.reshape(1,-1)
imp.fit(y2)
y_imp= imp.transform(y2)
X = array_imp[:,0:4]
Y = array_imp[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 7
scoring = 'accuracy'

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import  BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('BNB', BernoulliNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GBM', AdaBoostClassifier()))
models.append(('NN', MLPClassifier()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

