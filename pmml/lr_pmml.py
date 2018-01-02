import numpy as np
import pylab as pl
from  sklearn import linear_model,datasets

# import sample data from iris 
iris = datasets.load_iris()
X= iris.data[:,:2]
Y= iris.target

print(type(iris))

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X,Y)

# export model params
coefficients = logreg.coef_
intercept = logreg.intercept_
classes  = logreg.classes_
numberClasses=len(classes)
numberInputs=len(coefficients[0])

fo = open("RegressionModel.txt","w")
fo.write("LogisticRegression\n")
fo.write("IrisLogisticRegressionModel\n")
fo.write("classfication\n")
fo.write(str(numberInputs)+"\n")

'''
for num in range(0,numberInputs):
    fo.write(iris.feature_names[num]+",double,continuous,NA,NA,asMissing\n")
    
fo.write(str(numberClasses)+"\n")

for num in range(0,numberClasses):
    fo.write(iris.target_names[num]+"\n")
    
for num in range(0,numberClasses):
    fo.write(str(intercept[num])+"\n")

#for num in range(0,numberClasses):
    #for num2 in range(0,numberInputs):
        #fo.write(str(coefficients[num][num2])+"\n")

#fo.close()
'''