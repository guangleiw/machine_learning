import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

PATH = sys.path[0]

##### Acquire Data
train_src = pd.read_csv(PATH+"/data/train.csv")

#print(train_src.count())
#print(train_src.info())
#print(train_src.describe())
#print(train_src.describe(include=['0']))
#print(train_src.axes)

# value_counts 计算Series的每个值分布
#print(train_src['MSSubClass'].value_counts())
#print(train_src['MSZoning'].value_counts())

#LotFrontage 有空值
#print(train_src['LotFrontage'].value_counts())
#print(pd.isnull(train_src['LotFrontage']))

#print(train_src['LotArea'].value_counts())

#Alley的NA表示没有过道 不是指缺失值
#print(pd.isnull(train_src['Alley']))

#print(train_src['LotShape']);
#print(train_src.LotShape);
#print(train_src.LotShape[train_src['LotShape']);

## 统计LandContour为空的个数
#print((pd.isnull(train_src.LandContour)).value_counts())

#print(train_src.describe)
plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(train_src['SalePrice'],kde=False)
plt.xlabel("Sale Price")
plt.axes([0,800000,0,180])
plt.subplot(122)
sns.distplot(np.log(train_src['SalePrice']),kde=False)
plt.xlabel('Log (sale price)')
plt.axes([10,14,0,180])
plt.show()