from xml.dom import minidom
from pandas import DataFrame

from pandas import DataFrame
df=DataFrame([-1.959,5.238,5.172],index=['intercept','var_num1','var_num2'])


#pmml = minidom.parse('single_audit_logreg.pmml')
pmml = minidom.parse('lr.pmml')

root = pmml.documentElement

model = root.getElementsByTagName('GeneralRegressionModel')[0]
nameNodeList = model.getElementsByTagName("RegressionTable")

np = nameNodeList[0].getElementsByTagName("NumericPredictor")

for l in np:
    p=l.parentNode
    p.removeChild(l)


for row in df.itertuples():
    #print(type(row.Index))
    #print(row._1)
    if row.Index == 'intercept':
        nameNodeList[0].setAttribute('intercept',str(row._1))
        break;



for index,row in df.iterrows():
    if index == 'intercept':
        continue
    newEle = pmml.createElement("NumericPredictor")
    newEle.setAttribute("name",index)
    newEle.setAttribute("exponent","1")
    newEle.setAttribute("coefficient",str(row[0]))
    nameNodeList[0].appendChild(newEle)

f= open('rst.pmml', 'w')
root.writexml(f, addindent='  ', newl='\n')

#print(pmml.toxml())
#root.writexml(f, addindent='  ')
#root.close()
f.close()  


