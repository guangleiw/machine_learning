__author__ = 'wangguanglei1'

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data
iris=load_iris()

# create X(features) and y (response)
X=iris.data
y=iris.target

#use train/test split with different random_state values
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=5)

# check classification accuracy of KNN with K=5
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))

# we get different accuracy with different random_states

##### cross-validation example : parameter tuning
from sklearn.model_selection import cross_val_score

knn=KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
# print(scores)

#use average accurarcy as an estimate of out-of-sample accuracy
# print(scores.mean())

#search for an optional value of k for KNN
k_range=range(1,31)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
#print(k_scores)

import matplotlib.pyplot as plt
plt.plot(k_range,k_scores)
plt.xlabel('Value of k for  KNN')
plt.ylabel('Cross-Validated Accuracy')
#plt.show()
#####

##### cross-validation example : model selection
#compare the best KNN model with logistic regression on
#the iris dataset
#10-fold cross-validation with the best KNN model
knn=KNeighborsClassifier(n_neighbors=10)
#print(cross_val_score(knn,X,y,cv=10,scoring='accuracy').mean())

# 10-fold validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
#print(cross_val_score(logreg,X,y,cv=10,scoring='accuracy').mean())
#####


#####cross-validation : feature selection
#goal: select whether the Newspaper feature should be included
#in the linear regression model on the advertising dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# read in the advertising dataset
data = pd.read_csv('./data/Advertising.csv',index_col=0)

# create a python list of three feature names
feature_cols = ['TV','Radio','Newspaper']

#use the list to select a subset of the DataFrame(X)
X=data[feature_cols]

#select the Sales columns as the response(y)
y=data.Sales

#10-fold cross-validation with all three features
lm=LinearRegression()
scores=cross_val_score(lm,X,y,cv=10,scoring='neg_mean_squared_error')
# print(scores)

#fix the sign of MSE scores
mse_scores = -scores
rmse = np.sqrt(mse_scores)
# print(mse_scores)

#calculate the average RMSE
print(rmse.mean())

#remove the feature Newspaper
feature_cols=['TV','Radio']
X=data[feature_cols]
print(np.sqrt(-cross_val_score(lm,X,y,cv=10,scoring='neg_mean_squared_error')).mean())

#remove the feature Radio
feature_cols=['TV','Newspaper']
X=data[feature_cols]
print(np.sqrt(-cross_val_score(lm,X,y,cv=10,scoring='neg_mean_squared_error')).mean())