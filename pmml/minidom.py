from xml.dom import minidom
from pandas import DataFrame

from pandas import DataFrame
df=DataFrame([-1.959,5.238,5.172],index=['intercept','var_num1','var_num2'])

#print(df.index)


#pmml = minidom.parse('single_audit_logreg.pmml')
pmml = minidom.parse('lr.pmml')

root = pmml.documentElement

model = root.getElementsByTagName('GeneralRegressionModel')[0]
nameNodeList = model.getElementsByTagName("RegressionTable")
#nodeList = nameNode.getElementsByTagName("NumericPredictor")
#print("Total Numbers:"+len(nodeList))
#for ele in nodeList:
    #print(ele.attributes['name'].value)
    ##print(ele.attributes['label'].value)
    #ele.setAttribute('name','var_num1')
    #print(ele.attributes['name'].value)
    ##print(ele.attributes['label'].value)
    ##print(pmml.toxml())
    #break

for index,row in df.iterrows():
    #print(type(index[0]))
    #print(type(row[0]))
    newEle = pmml.createElement("NumericPredictor")
    newEle.setAttribute("name",index[0])
    newEle.setAttribute("exponent","1")
    newEle.setAttribute("coefficient",row[0])
    nameNodeList[0].appendChild(newEle)

print(pmml.toxml())
