import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns

PATH = sys.path[0]

##### Acquire Data
train_src = pd.read_csv(PATH+"data/train.csv")

print(train_src.head())