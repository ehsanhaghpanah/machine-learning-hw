
# %%

import pandas as pa
import seaborn as sb
import os
import inspect 
from IPython.display import display
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

#
def calc_lr(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool = False) -> tuple:
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
def calc_lasso_min(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool = False) -> tuple:
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

#
def calc_lasso_1se(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool = False) -> tuple:
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

#
def predict_lr(xTrainSet, yTrainSet, xTestSet, yTestSet, verbose: bool = False) -> tuple:
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

try:
     path: str = os.getcwd() + '\\data\\bitcoin-dataset.xlsx'
     data = pa.read_excel(path)

     # # 
     # display(data)

     # #
     # # we may consider {Daily Return} as dependent variable (in our model)
     # data['Daily Return'] = data['Bitcoin Market Price USD'].pct_change()
     # sb.displot(data['Daily Return'])

     # defining dependent variable and independent variables (vector)
     yName = 'Bitcoin Market Price USD' 
     y = data[yName]
     X = data.drop([yName], axis= 1)    # X is a vector so it is in capital

     # data normalization
     # we normalize data in order to make model parameters scale or unit agnostic
     X = (X - X.min()) / (X.max() - X.min())

     # number of features {k} has been chosen
     X = SelectKBest(f_regression, k= 5).fit_transform(X, y)

     # splitting (or folding) X
     # folding into {training set} and {validation set}
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state= 1)

     # calculating models parameters for comparison
     lr_train_score, lr_test_score = calc_lr(X_train, y_train, X_test, y_test)
     ridge_train_score, ridge_test_score = calc_ridge(X_train, y_train, X_test, y_test)
     lsmin_train_score, lsmin_test_score = calc_lasso_min(X_train, y_train, X_test, y_test)
     ls1se_train_score, ls1se_test_score = calc_lasso_1se(X_train, y_train, X_test, y_test)

except Exception as p:
     print(f'error {inspect.stack()[0][3]}, -> {p.args}')
     pass

# https://www.pluralsight.com/guides/linear-lasso-ridge-regression-scikit-learn


# %%
