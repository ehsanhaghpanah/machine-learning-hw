# %%

import pandas as pa
import seaborn as sb
import os
import inspect
from IPython.display import display

#
# loading data-set
try:
     path: str = os.getcwd() + '\\data\\bitcoin-dataset.xlsx'
     dataSet: pa.DataFrame = pa.read_excel(path)

     # data-set info
     print(f'DataSet Summey, {dataSet.shape}')
     display(dataSet)

     #
     # we may consider {Daily Return} as dependent variable (in our model)
     dataSet['Daily Return'] = dataSet['Bitcoin Market Price USD'].pct_change()
     sb.displot(dataSet['Daily Return'])

     print(dataSet['Daily Return'].max())
     print(dataSet['Daily Return'].min())

     # normalization and selecting response variable
     # defining dependent variable and independent variables (vector)
     yName = 'Bitcoin Market Price USD' 
     y = dataSet[yName]
     X = dataSet.drop([yName], axis= 1)    # X is a vector so it is in capital

except Exception as p:
     print(f'error {inspect.stack()[0][3]}, -> {p.args}')
     pass

# %%

try:

     # defining dependent variable and independent variables (vector)
     yName = 'Bitcoin Market Price USD' 
     y = dataSet[yName]
     X = dataSet.drop([yName], axis= 1)    # X is a vector so it is in capital

except Exception as p:
     print(f'error {inspect.stack()[0][3]}, -> {p.args}')
     pass
