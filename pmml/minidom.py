from xml.dom import minidom
from pandas import DataFrame

#pmml = minidom.parse('single_audit_logreg.pmml')
pmml = minidom.parse('lr.pmml')

root = pmml.documentElement

model = root.getElementsByTagName('GeneralRegressionModel')[0]
nameNode = model.getElementsByTagName("RegressionTable")[0]
nodeList = nameNode.getElementsByTagName("NumericPredictor")
#print("Total Numbers:"+len(nodeList))
for ele in nodeList:
    print(ele.attributes['name'].value)
    #print(ele.attributes['label'].value)
    ele.setAttribute('name','var_num1')
    print(ele.attributes['name'].value)
    #print(ele.attributes['label'].value)
    #print(pmml.toxml())
    break
    

from pandas import DataFrame
#d={'intercept':-1.959,'var_num1':5.238,'var_num2':5.172}
#print(type(d))
df=DataFrame([-1.959,5.238,5.172],index=['intercept','var_num1','var_num2'])
print(df.head(10))