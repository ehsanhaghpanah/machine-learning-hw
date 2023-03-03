# %%

import pandas as pa
import seaborn as sb
import os
import inspect
from IPython.display import display
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#
def calc_lsreg(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool = False) -> tuple:
     try:
          model: LinearRegression = LinearRegression().fit(xTrainSet, yTrainSet)
          
          score_train = model.score(xTrainSet, yTrainSet)
          score_test = model.score(xTestSet, yTestSet)

          if (verbose):
               print(f'beta0 = {model.intercept_}')
               print(f'beta* = {model.coef_}')
               print(f'score for trian set = {score_train}')
               print(f'score for test set  = {score_test}')

          return (score_train, score_test)
     except Exception as p:
          print(f'error {inspect.stack()[0][3]}, -> {p.args}')
          return (None, None)

#
def calc_ridge(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool = False) -> tuple:
     try:
          model: Ridge = Ridge().fit(xTrainSet, yTrainSet)

          score_train = model.score(xTrainSet, yTrainSet)
          score_test = model.score(xTestSet, yTestSet)

          if (verbose):
               print(f'beta0 = {model.intercept_}')
               print(f'beta* = {model.coef_}')
               print(f'score for trian set = {score_train}')
               print(f'score for test set  = {score_test}')

          return (score_train, score_test)
     except Exception as p:
          print(f'error {inspect.stack()[0][3]}, -> {p.args}')
          return (None, None)

#
def calc_lasso(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool = False) -> tuple:
     try:
          model: Lasso = Lasso().fit(xTrainSet, yTrainSet)
          
          score_train = model.score(xTrainSet, yTrainSet)
          score_test = model.score(xTestSet, yTestSet)

          if (verbose):
               print(f'beta0 = {model.intercept_}')
               print(f'beta* = {model.coef_}')
               print(f'score for trian set = {score_train}')
               print(f'score for test set  = {score_test}')

          return (score_train, score_test)
     except Exception as p:
          print(f'error {inspect.stack()[0][3]}, -> {p.args}')
          return (None, None)

# draws final results as a table
def drawFinalTable() -> None:
     dc: dict = {
          'Method': ['LS', 'Ridge', 'LASSO-Min', 'LASSO-1SE'],
          'Fit to Set 1': [0.0] * 4,
          'Fit to Set 2': [0.0] * 4
     }

     df: pa.DataFrame = pa.DataFrame(dc)
     display(df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

     # data normalization
     # we normalize data in order to make model parameters scale or unit agnostic
     X = (X - X.min()) / (X.max() - X.min())

     # number of features {k} has been chosen
     X = SelectKBest(f_regression, k= 5).fit_transform(X, y)

     # splitting X
     # splitting (or folding !?) into {traing set} and {validation set}
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state= 1)

     # calculating models parameters for comparison
     lsreg_train_score, lsreg_test_score = calc_lsreg(X_train, y_train, X_test, y_test)
     ridge_train_score, ridge_test_score = calc_ridge(X_train, y_train, X_test, y_test)
     lasso_train_score, lasso_test_score = calc_lasso(X_train, y_train, X_test, y_test)

except Exception as p:
     print(f'error {inspect.stack()[0][3]}, -> {p.args}')
     pass
