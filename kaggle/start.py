__author__ = 'wangguanglei1'
from sklearn.datasets import load_iris

iris = load_iris()
# print(type(iris))
# print(iris.data)
# print(iris.feature_names)
# print(iris.target)
# print(iris.target_names)
X=iris.data
y=iris.target

# print(X.shape)
# print(y.shape)

from sklearn.neighbors import KNeighborsClassifier

# we have a object called
knn=KNeighborsClassifier(n_neighbors=1)

# print(knn)

knn.fit(X,y)

outcome=knn.predict([3,5,4,2])
# print(outcome)

X_new = [[3,5,4,2],[5,4,3,2]]
outcome=knn.predict(X_new)
# print(outcome)

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
outcome = knn.predict(X)
# print(outcome)
from sklearn import metrics
print(metrics.accuracy_score(y,outcome))