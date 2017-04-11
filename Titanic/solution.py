import os
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid')

PATH=sys.path[0]
#print(PATH)

###
# Acquire Data
###
# download data from here:
# https://www.kaggle.com/c/titanic/data
train_src=pd.read_csv(PATH+"/data/train.csv")
test_src=pd.read_csv(PATH+"/data/test.csv")
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
old_age=train_src['Age'];
new_age=train_src['Age'].fillna(train_src['Age'].mean())
#print(old_age.describe());
#print(new_age.describe());
train_src['Age'] = new_age
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
print(train_src.shape)
print(test_src.shape)