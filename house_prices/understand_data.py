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
#sns.jointplot(x=var,y='SalePrice',data=data)
#sns.plt.show()
#与TotalBsmtSF的关系
var='TotalBsmtSF'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
#sns.jointplot(x=var,y='SalePrice',data=data)
#sns.plt.show()

#与categorical 变量的关系
var='OverallQual'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
#f,ax=plt.subplot(figure=(8,6))
#fig=sns.boxplot(x=var,y='SalePrice',data=data)
#sns.boxplot(x=var,y='SalePrice',data=data)
#fig.axis(ymin=0,ymax=800000)
#sns.plt.show()

var='YearBuilt'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
#fig=sns.boxplot(x=var,y='SalePrice',data=data)
#fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
#sns.plt.show()

########################################
# 客观分析 -- 相关矩阵
corrmat=df_train.corr()
#fig=sns.heatmap(corrmat,vmax=.8,square=True)
# 设置坐标展示方向 避免重叠
#fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
#fig.set_yticklabels(fig.get_yticklabels(),rotation=30)
#sns.plt.show()

# 客观分析 -- SalePrice的相关矩阵
# 取出和 SalePrice相关性前10的变量
K=10
cols=corrmat.nlargest(K,'SalePrice')['SalePrice'].index
cm=np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
#hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
#hm.set_xticklabels(hm.get_xticklabels(),rotation=90)
#hm.set_yticklabels(hm.get_yticklabels(),rotation=30)
#sns.plt.show()

# 客观分析 -- SalePrice和相关变量的散点图
sns.set()
cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(df_train[cols],size=1.5)
#sns.plt.show()

########################################
# 缺失值处理
total=df_train.isnull().sum().sort_values(ascending=False)
percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
#print(missing_data.head(20))

#deleting the missing data
df_train= df_train.drop((missing_data[missing_data['Total']>1]).index,1)
df_train=df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
#print(df_train.isnull().sum().max())

########################################
# 奇异值处理
# standardizing the data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
#print('outer range (low) of the distribution:')
#print(low_range)
#print('\nouter range (high) of the distribution:')
#print(high_range)

# 二元变量的散点图 -- 深入
var='GrLivArea'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
#data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#sns.plt.show()

#delete points
df_train.sort_values(by='GrLivArea',ascending=False)[:2]
df_train=df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train=df_train.drop(df_train[df_train['Id'] == 524].index)

var='TotalBsmtSF'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
#data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#sns.plt.show()

#histogram and normal probability plot
#sns.distplot(df_train['SalePrice'],fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['SalePrice'],plot=plt)
#sns.plt.show()

#applying log transformation
sns.distplot(np.log(df_train['SalePrice']),fit=norm)
fig = plt.figure()
res = stats.probplot(np.log(df_train['SalePrice']),plot=plt)
sns.plt.show()
