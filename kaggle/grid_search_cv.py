import sklearn
#print(sklearn.__version__)
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# GridSearchCV is a more efficient  way tuning parameter
# Allows you to define a grid of parameters that will be searched using K-fold cross-validation

# read in the iris data
iris=load_iris()
# create X(features) and y (response)
X=iris.data
y=iris.target

# define the parameter values that should be searched
k_range = range(1,31)
#print(k_range)

# create a parameter grid: map the parameter names to the values that should be searched
param_grid =dict(n_neighbors=k_range)
#print(param_grid)

knn=KNeighborsClassifier()

# instantiate the grid
# grid does not include data set X and y , but it does include param_grid
# grid is an object that is ready to do 10-fold cross validation on an KNN model
# using classification_accuracy as the evaluation metic.
# But in addition it is give this param_grid , so it knows that the cross-validtion process 
# will be repeated 30 times and each time the 'n_neighbors' parameter will be given a different value 
# from the list. That is why the param_grid is specified using key-value pairs.
grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')

# fit the grid with data
grid.fit(X,y)

# view the complete result
## Warning: 
# The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute.
#print(grid.grid_scores_)
# examine the 1st tuple
#print (grid.grid_scores_[0].parameters)
#print (grid.grid_scores_[0].cv_validation_scores)
#print (grid.grid_scores_[0].mean_validation_score)
##
## using grid.cv_results_
#print(grid.cv_results_['mean_train_score'])
#print(grid.cv_results_['mean_test_score'])
##

#plot the result
import matplotlib.pyplot as plt
plt.plot(k_range,grid.cv_results_['mean_test_score'])
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-validation Accuracy')
plt.show()

# Notice: best_* is the properties of grid , not cv_results_
print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)
