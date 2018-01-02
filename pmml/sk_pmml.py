import pandas
from sklearn import datasets

#iris_df = pandas.read_csv("Iris.csv")
iris_df = datasets.load_iris()

from sklearn2pmml import PMMLPipeline
from sklearn.tree import DecisionTreeClassifier

iris_pipeline = PMMLPipeline([("classifier",DecisionTreeClassifier())])

iris_pipeline.fit(iris_df[iris_df.columns.difference(["Species"])], iris_df["Species"])