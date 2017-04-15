import os
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
sns.set(style='whitegrid')

PATH=sys.path[0]
#print(PATH)

###
# Acquire Data
###
# download data from here:
# https://www.kaggle.com/c/titanic/data
train_src=pd.read_csv(PATH+"/input/train.csv")
test_src=pd.read_csv(PATH+"/input/test.csv")
#print(test_src.head())

###
# Analyze by describing data
# Q: You should identify which features are numerical(discrete / continous) , categorical and mixed data types 
# A: categorical: survived sex embarked; ordinal: Pclass
# A: numerical: continous (age fare) discrete(sibsp parch)
# Q: Which features are mixed data types?
# A: Ticket is a mix of numerical and alphanumeric data types; Cabin is alphanumeric
### 
# Q: Whice features may contain errors or typos?
# A: This is harder to review for a large ele, 
# A: however reviewing a few samples from a smaller ele may just tell us outright, 
# Q: which features may require correcting.
# A: Name feature may contain errors or typos as there are several ways used to describe a name including titles, 
# A: round brackets, and quotes used for alternative or short names.
# Q: Which features contain blank, null or empty values?
# A: Cabin > Age > Embarked (training data) Cabin > Age are incomplete of test ele
# Q: What are the data types for various features?
###
#print(type(train_src))
#print(train_src.axes)
#print(train_src.head())
#print(train_src.info())
#print('+'*40)
#print(train_src.info())
#print(train_src.head())
#print(train_src.describe(include=['O']))
#print(train_src.info())
#exit()

#print(train_src['Age']);
#print(pd.isnull(train_src['Age']))
#old_age=train_src['Age'];
#new_age = old_age;
#new_age=train_src['Age'].fillna(train_src['Age'].mean())
#print(old_age.describe());
#print(new_age.describe());
#train_src['Age'] = new_age
#print(pd.isnull(train_src['Age']))
#exit()

###
#S0=train_src.Age[train_src.Survived ==0].value_counts()
#S1=train_src.Age[train_src.Survived ==1].value_counts()

#fig = plt.figure()
#fig.set(alpha=0.2)
#df=pd.DataFrame({'saved':S1,'dead':S0})
#df.plot(kind='bar',stacked=True)
#plt.show()
#exit()
###
#g=sns.FacetGrid(train_src,col='Survived')
#g.map(plt.hist,'Age',bins=20)
#sns.plt.show()

#g=sns.factorplot(x='Age',data=train_src,kind='count')
#sns.plt.show()

#g=sns.factorplot("Survived",col='Pclass',data=train_src,kind='count',aspect=.8)
#sns.plt.show()

#print(train_src['SibSp'].describe())
#g=sns.factorplot("Survived",col="SibSp",col_wrap=3,data=train_src,kind='count')
#sns.plt.show()

#g=sns.factorplot("Survived",col="Parch",col_wrap=3,data=train_src,kind="count")
#sns.plt.show()

#print(train_src['Fare'].describe())


## 对于离散型的（性别 等级 ） 通过计算计算概率 当然也可以画图
sex_cor=train_src[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
pclass_cor=train_src[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
parch_cor = train_src[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)

## 对于连续型的数值 划分band 画出柱状图 check 分布状况 更直观
#g=sns.FacetGrid(train_src,col='Survived')
#g.map(plt.hist,'Age',bins=20)
#g.map(plt.hist,'Fare',bins=20)
#sns.plt.show()

#grid = sns.FacetGrid(train_src, col='Pclass', row='Survived', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend();
#sns.plt.show()


#fp = sns.factorplot('Pclass',hue='Survived',data=train_src,size=3,kind='count')
fp = sns.factorplot('Sex',hue='Survived',data=train_src,size=3,kind='count')
#fp.despine(left= True)
#sns.plt.show()

## drop features
train_src = train_src.drop(['Ticket','Cabin'],axis=1)
test_src = test_src.drop(['Ticket','Cabin'],axis=1)
#print(train_src.Name)
#all_src = pd.DataFrame([train_src,test_src])
#all_src = train_src.append(test_src)
#print(all_src.describe())
all_src = [train_src,test_src]
#print(type(all_src))
#exit();

for ele in all_src:
	ele['Title'] = ele.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
	#print(ele['Title'])

#print(all_src[1].Title)
pd.crosstab(train_src['Title'],train_src['Sex'])

for ele in all_src:
	ele['Title'] = ele['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
	ele['Title'] = ele['Title'].replace('Mlle', 'Miss')
	ele['Title'] = ele['Title'].replace('Ms', 'Miss')
	ele['Title'] = ele['Title'].replace('Mme', 'Mrs')

## check the distribution of title
#print(train_src[['Title','Survived']].groupby(['Title'],as_index=False).mean())

# replace the Name to ordinal
title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}
for ele in all_src:
	ele['Title'] = ele['Title'].map(title_mapping)
	ele['Title'] = ele['Title'].fillna(0)
	
#print(train_src.head())

#drop the Name and PassengerId
train_src = train_src.drop(['Name','PassengerId'],axis=1)
test_src = test_src.drop(['Name'],axis=1)
all_src = [train_src,test_src]
#print(train_src.shape)
#print(test_src.shape)

# Converting a categorical feature gender
for ele in all_src:
	ele['Sex'] = ele['Sex'].map({"male":0,"female":1}).astype(int)

#print(train_src.head())

guess_ages = np.zeros((2,3))
for ele in all_src:
	#print(ele.info());exit();
	for i in range(0,2):
		for j in range(0,3):
			idx = (ele['Sex'] == i) & (ele['Pclass'] == j+1);
			guess = ele[idx]['Age'].dropna()
			#print(guess);exit();
			guess_age = guess.median()
			#print(type(guess_age)) 
			#exit()			
			guess_ages[i,j] = (guess_age/0.5 + 0.5)*0.5
			
	for i in range(0,2):
		for j in range(0,3):
			ele.loc[(ele.Age.isnull())&(ele['Sex'] == i)&(ele['Pclass'] == j+1),'Age'] = guess_ages[i,j]
	ele['Age'] = ele.Age.astype(int)

#print(train_src.head())
# Create Age bands and determine correlations with Survived.

train_src['AgeBand'] = pd.cut(train_src['Age'],5)
age_sur = train_src[['AgeBand','Survived']].groupby(['AgeBand'],as_index = False).mean().sort_values(by='AgeBand',ascending=True)
#print(age_sur)

for ele in all_src:
	ele.loc[ele['Age']<= 16,'Age'] = 0
	ele.loc[(ele['Age']> 16) & (ele['Age'] <=32),'Age'] = 1
	ele.loc[(ele['Age']> 32) & (ele['Age'] <=48),'Age'] = 2
	ele.loc[(ele['Age']> 48) & (ele['Age'] <=64),'Age'] = 3
	ele.loc[ele['Age']>64,'Age'] = 4
	
#print(train_src.head())
# remove the AgeBand feature
train_src = train_src.drop(['AgeBand'],axis = 1)
all_src = [train_src,test_src]
#print(train_src.head())


# create a new feature combining existing fetures
# we can create a new feature for FamilySize which combines Parch and SibSp
# And then , we can drop the above features
for ele in all_src:
	ele['FamilySize'] = ele['SibSp'] + ele['Parch'] + 1
famsize_sur = train_src[['FamilySize','Survived']].groupby(['FamilySize'],as_index = False).mean().sort_values(by='Survived',ascending=False)
#print(famsize_sur)

#create a new feature called IsAlone
for ele in all_src:
	ele['IsAlone'] = 0;
	ele.loc[ele.FamilySize == 1,'IsAlone'] = 1

#print(train_src[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean())

#Let us drop Parch SibSp and FamilySize features in favor of IsAlone
train_src  =train_src.drop(['Parch','SibSp','FamilySize'],axis=1)
test_src = test_src.drop(['Parch','SibSp','FamilySize'],axis=1)
all_src = [train_src,test_src]

#print(train_src.head())

###Completing a categorical feature
# Embarked feature takes S Q C values based on port of embarkation .
# Our training dataset has two missing values.
# we simply fill these 	with the most common occurance 
freq_port = train_src.Embarked.dropna().mode()[0]
print(freq_port)
for ele in all_src:
	ele['Embarked'] = ele['Embarked'].fillna(freq_port)
#print(train_src[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False))
#converting categorical feature to numeric
for ele in all_src:
	ele['Embarked'] = ele['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
#print(train_src.head())

# Complete the Fare feature for single missing value in
# test dataset using mode to get the value 
# that occurs most frequently for this feature
# we do this in a line of code 
test_src['Fare'].fillna(test_src['Fare'].dropna().median(),inplace = True)
#print(test_src.head())
# we can also create fareband
train_src['FareBand'] = pd.qcut(train_src['Fare'],4)
fareband_sur = train_src[['FareBand','Survived']].groupby(['FareBand'],as_index = False).mean().sort_values(by='FareBand',ascending=True)
#print(fareband_sur);

for ele in all_src:
	ele.loc[ele['Fare']<=7.91,'Fare'] = 0
	ele.loc[(ele['Fare']>7.91) & (ele['Fare']<=14.454),'Fare'] = 1
	ele.loc[(ele['Fare']>14.454) & (ele['Fare']<= 31),'Fare'] = 2
	ele.loc[(ele['Fare']>31) & (ele['Fare']<= 512.329),'Fare'] = 3
	ele['Fare'] = ele['Fare'].astype(int)
	
train_src = train_src.drop(['FareBand'],axis = 1)
all_src = [train_src , test_src]
#print(train_src.head())
#print(test_src.head(10))


##########
#Model  predict  and solve
# Now we are going to train a model and predict the required solution
# There are 60+ predictive modelling algorithms to choose from.
# We must understand the type of problem and solution requirement to narrow down to a select few models
# which we can evaluate. 
# Our problem is a classification and regression problem.
# We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). 
# We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset.
# With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few.
## These include :
# Logistic Regression
# KNN or k-Nearest Neighbors
# Support Vector Machines
# Naive Bayes classifier
# Decision Tree
# Random Forrest
# Perceptron
# Artificial neural network
# RVM or Relevance Vector Machine

#prepare data
X_train = train_src.drop("Survived",axis=1)
#Y_train = train_src.Survived
Y_train = train_src['Survived']
X_test = test_src.drop("PassengerId",axis=1).copy()
#print(X_train.shape)
#print(Y_train.shape)
#print(X_test.shape)

# Logsitic Regression
lreg = LogisticRegression();
lreg.fit(X_train,Y_train)
Y_pred = lreg.predict(X_test)
lreg_acc = round(lreg.score(X_train,Y_train)*100,2)
#print(lreg_acc)

coeff_df = pd.DataFrame(train_src.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(lreg.coef_[0])

#print(coeff_df.sort_values(by='Correlation',ascending=False))


# Support Vector Machines
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
svc_acc = round(svc.score(X_train,Y_train)*100,2)
#print(svc_acc)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)
knn_acc = round(knn.score(X_train,Y_train)*100,2)
#print(knn_acc)

# Gaussian Naive Bayes
gaus = GaussianNB()
gaus.fit(X_train,Y_train)
Y_pred = gaus.predict(X_test)
gaus = round(gaus.score(X_train,Y_train)*100,2)
#print(gaus)

# Perceptron
# Linear SVC
# Stochastic Gradient Descent

# Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train,Y_train)
Y_pred = dec_tree.predict(X_test)
dec_acc = round(dec_tree.score(X_train,Y_train)*100,2)
# print(dec_acc)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
rad_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(rad_forest)

########
# Model evaluation 
# we can now rank our evaluation of all the models to choose 
# the best one for our problem.
# while both DEcision Tree and Random Forest score the same.
# we choose to use random forest as they correct for decision trees
# habit of overfitting to their training set
models = pd.DataFrame({'Model':['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Decision Tree'],'Score':[svc_acc,knn_acc,lreg_acc,rad_forest,dec_acc]})
#print(models.sort_values(by='Score',ascending=False))


submission = pd.DataFrame({"PassengerId":test_src["PassengerId"],"Survived":Y_pred})
submission.to_csv(PATH+"/output/submission.csv",index=False)



