from sklearn.datasets import load_iris
from sklearn.grid_search import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Searching many different parameters at once may be computationally infeasible
# RandomizedSearchCV searches a subset of the parameters , and you control the computational 'budget'

iris = load_iris();
X=iris.data
y=iris.target
k_range = range(1,31)
weight_options = 'uniform'

# specify "parameter distributions " rather than a "parameter grid"
param_dist = dict(n_neighborts=k_range,weights=weight_options)
knn=KNeighborsClassifier(n_neighbors=13,weights='uniform')
rand = RandomizedSearchCV(knn,param_dist,cv=10,scoring='accuracy',n_iter=10,random_state=5)
rand.fit(X,y)
print(rand.grid_scores_)

