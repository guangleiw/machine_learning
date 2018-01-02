import pandas as pd
url='http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima=pd.read_csv(url,header=None,names=col_names)

#print the first 5 rows of data
#print(pima.head())

###
# Our target: predict the diabetes status of a patient given their health measurements

#define X and y
feature_cols= ['pregnant','insulin','bmi','age']
X=pima[feature_cols]
y=pima.label

# Split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)

#   make class predictions for the testing set
y_pred_class=logreg.predict(X_test)
#print(type(y_pred_class))
###

###
# Caculate the accuracy: percentage of correct predictions
from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred_class)

# You dont know the underlying distribution of response values and 
# you dont know the 'types' of errors your classifier is making
# Use Null accuracy to solve the problems

# examine the class distribution of the testing set(using a pandas series method)
# calculate the count of the value in y_test
#print(y_test.value_counts()) 

# calculate the percentage of ones / zeros
#print(y_test.mean())
#print(1-y_test.mean())

# calculate null accuracy (for binary classification problems coded as 0/1)
#print(max(y_test.mean(),1-y_test.mean()))

# calculate null accuracy (for multi-class classification problems)
#print(y_test.value_counts().head(1)/len(y_test))

# Comparing the ture and predicted response values
#   print the first 25 true and predicted responses
#print('True:',y_test.values[0:25])
#print('Pred:',y_pred_class[0:25])

###

# Confusion metrics
confusion = metrics.confusion_matrix(y_test,y_pred_class)
#print('True:',y_test.values[0:25])
#print('Pred:',y_pred_class[0:25])

TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]

total = float(TP+TN+FP+FN)

# classification Accuracy: overall , how often is the classification correct?
#print((TP+TN)/total)
#print(metrics.accuracy_score(y_test,y_pred_class))

# classification error: overall , how often is the classifier incorrect?
#print((FP+FN)/total)
#print(1-metrics.accuracy_score(y_test,y_pred_class))

# sensitivity: When the actual value is positive , how often is the prediction correct?
# how 'sensitive' is the classifier to detecting positive instances
# also known as 'True Positive Rate' or 'Recall'
#print(TP/float(TP+FN))
#print(metrics.recall_score(y_test,y_pred_class))

# specificity: when the actual value is negative , how often is the prediction correct?
# how specific is the classifier in predicting positive instances?
#print(TN/float(TN+FP))

# precision: Wnen a positive value is predicted , how often is the predicition correct?
# how 'precise' is the classifier when predicting positive instances?
#print(TP/(float(TP+FP)))
#print(metrics.precision_score)

###
###Adjusting the classification threshold
#print(logreg.predict(X_test)[0:10])
# print the first 10 predicted porbability of class membership
# 这个地方的概率是？
# 回想一下 0 和 1 并不是预测的直接结果 而是由一个概率值+判定阈值来决定的是0还是1
# 而predict_proba 就是预测的概率值
print(logreg.predict_proba(X_test)[0:10 , :])
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# 
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

#histogram of predicted probabilities
#plt.hist(y_pred_prob,bins=8)
#plt.xlim(0,1)
#plt.title('Histogram of predicted probabilities')
#plt.xlabel('Predicted probabilities of diabetes')
#plt.ylabel('Frequency')
#plt.show()

#DECREASE the THRESHOLD for predicting diabetes in order to increase the sensitivity of the classifiter
# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize(y_pred_prob,0.3)[0] #y_pred_class is determined by y_pred_prob and threshold

#print(y_pred_prob[0:10])
#print(y_pred_class[0:10])
#print(confusion)
new_confusion = metrics.confusion_matrix(y_test,y_pred_class)
#print(new_confusion)

## ROC Curves
# 那么问题来了,到底threshold该取哪一个值呢？
# 一个一个试 会飞起来的
# 如果能有一条曲线 能标明sensitivity (specificity) 与 threshold的关系 那就好了

#important： first argument is true values, second argument is predicted probabilities
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

# We cannot see the threshold from the ROC Curve
