import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm 
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

df_train=pd.read_csv('data/train.csv')
#print(df_train.columns)

########################################
# Understand SalePrice 
print(df_train['SalePrice'].describe())
sns.distplot(df_train['SalePrice'])
#sns.plt.show()

# 偏离度
print("Skewness: %f"% df_train['SalePrice'].skew())
# 峰度
print("Kurtosis: %f" %df_train['SalePrice'].kurt())

########################################
# SalePrice 与其它变量的关系
# 与numerical的关系
var='GrLivArea'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
#不知道为什么用data.plot.scatter方法显示不出图像
#data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
sns.jointplot(x=var,y='SalePrice',data=data)
#sns.plt.show()
#与TotalBsmtSF的关系
var='TotalBsmtSF'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
sns.jointplot(x=var,y='SalePrice',data=data)
#sns.plt.show()

#与categorical 变量的关系
var='OverallQual'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
#f,ax=plt.subplot(figure=(8,6))
#fig=sns.boxplot(x=var,y='SalePrice',data=data)
sns.boxplot(x=var,y='SalePrice',data=data)
#fig.axis(ymin=0,ymax=800000)
sns.plt.show()