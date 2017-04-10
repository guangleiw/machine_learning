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
# A: This is harder to review for a large dataset, 
# A: however reviewing a few samples from a smaller dataset may just tell us outright, 
# Q: which features may require correcting.
# A: Name feature may contain errors or typos as there are several ways used to describe a name including titles, 
# A: round brackets, and quotes used for alternative or short names.
# Q: Which features contain blank, null or empty values?
# A: Cabin > Age > Embarked (training data) Cabin > Age are incomplete of test dataset
# Q: What are the data types for various features?
###
#print(type(train_src))
#print(train_src.axes)
#print(train_src.tail())
#print(train_src.info())
#print('+'*40)
#print(train_src.info())
#print(train_src.head())
#print(train_src.describe(include=['O']))

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
g=sns.factorplot("Survived",col="SibSp",col_wrap=3,data=train_src,kind='count')
sns.plt.show()


