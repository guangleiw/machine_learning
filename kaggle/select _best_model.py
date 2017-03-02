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
from sklearn.cross_validation import cross_val_score

knn=KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
print(scores)

#use average accurarcy as an estimate of out-of-sample accuracy
print(scores.mean())

#search for an optional value of k for KNN
k_range=range(1,31)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

import matplotlib.pyplot as plt
plt.plot(k_range,k_scores)
plt.xlabel('Value of k for  KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()