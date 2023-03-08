
# %%

import pandas as pa
# import seaborn as sb
import os
import inspect

from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#
def calc_lsreg(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool= False) -> tuple:
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
def calc_ridge(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool= False) -> tuple:
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
def calc_lasso(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool= False) -> tuple:
     try:
          model: Lasso = Lasso(tol= 1e-2).fit(xTrainSet, yTrainSet)
          
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
def drawFinalTable(lsreg: tuple, ridge: tuple, lasso: tuple) -> None:
     dc: dict = {
          'Method': ['LS', 'Ridge', 'LASSO'],
          'Fit to Set 1': [lsreg[0], ridge[0], lasso[0]],
          'Fit to Set 2': [lsreg[1], ridge[1], lasso[1]]
     }

     df: pa.DataFrame = pa.DataFrame(dc)
     display(df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main_run(verbose: bool= False) -> None:
     try:
          path: str = os.getcwd() + '\\data\\bitcoin-dataset.xlsx'
          dataSet: pa.DataFrame = pa.read_excel(path)

          # data-set info
          if (verbose):
               print(f'DataSet Summey, {dataSet.shape}')
               display(dataSet)

          # {Date} field must be removed because of {StandardScaler}
          X = dataSet.drop(['Date'], axis= 1)

          # defining dependent variable and independent variables (vector)
          yName = 'Bitcoin Market Price USD' 
          y = X[yName]                       # y is a vector (or series) so it is in lower-case
          X = X.drop([yName], axis= 1)       # X is a data-frame so it is in upper-case

          # splitting (or folding !?) into {traing set} and {validation set}
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state= 1)

          # normalization and standardization
          # NOTE: standardization is applied on train set, 
          # neither the whole data-set nor the test-set
          scaler = StandardScaler().fit(X_train)
          X_train_stan = scaler.transform(X_train)
          X_test_stan  = scaler.transform(X_test)

          # output
          if (verbose):
               print(f'X Standard Train Set: Features = {X_train_stan.shape[1]}, Samples = {X_train_stan.shape[0]}')
               print(f'X Standard Test Set:  Features = {X_test_stan.shape[1]}, Samples = {X_test_stan.shape[0]}')

          # splitting X
          # splitting (or folding !?) into {traing set} and {validation set}
          # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state= 1)
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5)

          # calculating models parameters for comparison
          lsreg_train_score, lsreg_test_score = calc_lsreg(X_train, y_train, X_test, y_test)
          ridge_train_score, ridge_test_score = calc_ridge(X_train, y_train, X_test, y_test)
          lasso_train_score, lasso_test_score = calc_lasso(X_train, y_train, X_test, y_test)

          # printing out
          drawFinalTable(
               (lsreg_train_score, lsreg_test_score),
               (ridge_train_score, ridge_test_score),
               (lasso_train_score, lasso_test_score)
          )

     except Exception as p:
          print(f'error {inspect.stack()[0][3]}, -> {p.args}')
          pass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

main_run()

# %%
