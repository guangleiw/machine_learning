__author__ = 'wangguanglei1'

#conversional way to import pandas
import seaborn as sns
import pandas as pd
import numpy as np



# read data from URL ans saved the data
data=pd.read_csv('D://machine_learning//kaggle//data//Advertising.csv')

# display the 1st 5 head lines
# print(data.head())
# print(data.columns)

#check the shape of the dataframe(rows,columns)
# print(data.shape)

## Visualizing the relationshop between features and the response using scatterplots
# sns.set(style="ticks", color_codes=True)
# sns.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',size=7,aspect=0.7,kind='reg')
#dont forget this function
#%matplotlib inline > displays the plots INSIDE the notebook
#sns.plt.show() > displays the plots OUTSIDE of the notebook
# sns.plt.show()

# create a Python list of feature names
feature_cols=['TV','Radio','Newspaper']

#use the list to select a subset of the orginal Dataframe
X=data[feature_cols]
# print(X.head())
# print(type(X))
# print(X.shape)

# y=data['Sales']
y=data.Sales
# print(y.head())
# print(type(y))

##### Splitting X and y into training and testing sets
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

#default split is 75% for training and 25% for testing
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


##### Linear regression in scikit-learn
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train,y_train)

#pair the feature names with the coefficients
#For python3.0 users , zip() returns an object
#have to use list to convert
z=zip(feature_cols,linreg.coef_)
# print(list(z))

#interpreting the model coefficients
# print(linreg.intercept_)
# print(linreg.coef_)

##### Making predictions
y_pred=linreg.predict(X_test)

##### Computing the RMSE(Root Mean Squared Error) for our Sales predictions
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))