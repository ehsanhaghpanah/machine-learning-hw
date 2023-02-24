#
# Copyright (C) Remittan (remittan.com), 2023.
# All rights reserved.
# Ehsan Haghpanah; haghpanah@remittan.com
#

# %%

import pandas as pa
from IPython.display import display
import pandas as pa
import numpy as np
import seaborn as sb
import matplotlib.pyplot as pl
import inspect
import os
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

#
def getDataSet(path: str = '') -> pa.DataFrame:
     try:
          if (path == ''):
               path = (os.getcwd() + '\\data\\bitcoin-dataset.xlsx')
          data = pa.read_excel(path)

          # defining dependent variable and independent variables (vector)
          yName = 'Bitcoin Market Price USD' 
          y = data[yName]
          X = data.drop([yName], axis= 1)    # X is a vector so it is in capital

          # data normalization
          X = (X - X.min()) / (X.max() - X.min())

          # number of features {k} has been set
          return SelectKBest(f_regression, k= 5).fit_transform(X, y)

          # # splitting X
          # # splitting into {traing set} and {validation set}
          # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state= 1)

     except Exception as p:
          print(f'error {inspect.stack()[0][3]}, -> {p.args}')
          pass

# draws final results as a table
def drawFinalTable() -> None:
     dc: dict = {
          'Method': ['LS', 'Ridge', 'LASSO-Min', 'LASSO-1SE'],
          'Fit to Set 1': [0.0] * 4,
          'Fit to Set 2': [0.0] * 4
     }

     df: pa.DataFrame = pa.DataFrame(dc)
     display(df)

X = getDataSet()

# %%

# %%
