# %%

import pandas as pa
import numpy as np
import matplotlib.pyplot as pl
import os
import inspect

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

from IPython.display import display

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Ridge (Modeling)
# returns MSPE for provided args
def calc_ridge(validator: RepeatedKFold, trX: pa.DataFrame, tsX: pa.DataFrame, trY: pa.Series, tsY: pa.Series) -> np.float64:
     """ calculating Ridge MSPE """
     try:

          # alpha = lambda ** (-1)
          cv = RidgeCV(alphas= np.arange(0.5, 2, 0.01), cv= validator, scoring= 'neg_mean_squared_error')
          cv.fit(trX, trY)    # providing X train set as well as y train set (or series)

          # alpha is known here using cross-validation 
          model = Ridge(alpha= cv.alpha_)
          model.fit(trX, trY)
          guess = model.predict(tsX)
          error = mean_squared_error(tsY, guess)

          return error
     except Exception as p:
          print(f'error {inspect.stack()[0][3]}, -> {p.args}')
          raise p

# Lasso (Modeling)
# returns MSPE for provided args
def calc_lasso(validator: RepeatedKFold, trX: pa.DataFrame, tsX: pa.DataFrame, trY: pa.Series, tsY: pa.Series) -> tuple:
     """ calculating Lasso MSPE """
     try:

          # alpha = lambda ** (-1)
          cv = LassoCV(alphas= np.arange(0.5, 2, 0.01), cv= validator, max_iter= 1000)
          cv.fit(trX, trY)    # providing X train set as well as y train set (or series)

          # alpha is known here using cross-validation 
          model = Lasso(alpha= cv.alpha_)
          model.fit(trX, trY)
          guess = model.predict(tsX)
          error = mean_squared_error(tsY, guess)

          return (error, len(model.coef_) - sum(model.coef_ == 0))
     except Exception as p:
          print(f'error {inspect.stack()[0][3]}, -> {p.args}')
          raise p

pcr_nfeat = []; pcr_mspes = []
pls_nfeat = []; pls_mspes = []

# PCR (Modeling)
# returns MSPE for provided args
def calc_pcr(validator: RepeatedKFold, trX: pa.DataFrame, tsX: pa.DataFrame, trY: pa.Series, tsY: pa.Series, featureCount: int, foldIndex: int) -> None:
     """ calculating PCR MSPE """
     try:
          
          pca = PCA()
          model = LinearRegression()
          for trIndex, tsIndex in validator.split(trX):
               
               print(f'PCR modeling Cross-Validation Index = {trIndex}')

               _trX, _tsX = trX[trIndex], trX[tsIndex]
               _trY, _tsY = trY.iloc[trIndex], trY.iloc[tsIndex]
               
               trPCs = pca.fit_transform(_trX)
               tsPCs = pca.transform(_tsX)
               mspes = []
               for featureIndex in range(1, featureCount):
                    model.fit(trPCs[:, : featureIndex], _trY)
                    guess = model.predict(tsPCs[:, : featureIndex])
                    error = mean_squared_error(_tsY, guess)
                    mspes.append(error)

          pcr_nfeat.append(np.argmin(mspes))
          trPCs = pca.fit_transform(trX)
          tsPCs = pca.transform(tsX)
          
          model.fit(trPCs[:, : pcr_nfeat[foldIndex]], trY)
          guess = model.predict(tsPCs[:, : pcr_nfeat[foldIndex]])
          error = mean_squared_error(tsY, guess)

     except Exception as p:
          print(f'error {inspect.stack()[0][3]}, -> {p.args}')
          raise p

# PLS (Modeling)
# returns MSPE for provided args
def calc_pls(validator: RepeatedKFold, trX: pa.DataFrame, tsX: pa.DataFrame, trY: pa.Series, tsY: pa.Series, featureCount: int, foldIndex: int) -> None:
     """ calculating PLS MSPE """
     try:
          
          # model = PLSRegression()
          # for trIndex, tsIndex in validator.split(trX):
          #      _trX, _tsX = trX[trIndex], trX[tsIndex]
          #      _trY, _tsY = trY.iloc[trIndex], trY.iloc[tsIndex]
               
          #      trPCs = pca.fit_transform(_trX)
          #      tsPCs = pca.transform(_tsX)
          #      mspes = []
          #      for featureIndex in range(1, featureCount):
          #           model.fit(trPCs[:, : featureIndex], _trY)
          #           guess = model.predict(tsPCs[:, : featureIndex])
          #           error = mean_squared_error(_tsY, guess)
          #           mspes.append(error)

          # pcr_nfeat.append(np.argmin(mspes))
          # trPCs = pca.fit_transform(trX)
          # tsPCs = pca.transform(tsX)
          
          # model.fit(trPCs[:, : pcr_nfeat[foldIndex]], trY)
          # guess = model.predict(tsPCs[:, : pcr_nfeat[foldIndex]])
          # error = mean_squared_error(tsY, guess)

          pass
     except Exception as p:
          print(f'error {inspect.stack()[0][3]}, -> {p.args}')
          raise p

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

try:
     path: str = os.getcwd() + '\\data\\bitcoin-dataset.xlsx'
     rawData: pa.DataFrame = pa.read_excel(path)

     # NOTE: {Date} field must be removed because of {StandardScaler}, 
     # which may result in wrong conclusion!
     dataSet: pa.DataFrame = rawData.drop(['Date'], axis= 1)

     # NOTE: this type of cross-validation may not be suitable for 
     # this data-set as the original data-set is a time-series
     totalRecords = dataSet.shape[0]
     foldCount = 10
     crossValidation = np.random.randint(low= 0, high= foldCount, size= totalRecords)
     dataSet['CV'] = crossValidation

     # display(dataSet)
     # print(dataSet['CV'].value_counts())
     # print(dataSet.isnull().sum())

     lsreg_mspes = []
     ridge_mspes = []
     lasso_mspes = []
     lasso_nfeat = []

     # calculation
     for foldIndex in range(foldCount):

          print(f'calculating for fold index = {foldIndex}')

          dataSet_tr = dataSet.loc[dataSet['CV'] != foldIndex]   # train data-set
          dataSet_ts = dataSet.loc[dataSet['CV'] == foldIndex]   # test data-set

          # defining dependent variable and independent variables (vector)
          yName = 'Bitcoin Market Price USD' 
          y_tr = dataSet_tr[yName]      # train data-set
          y_ts = dataSet_ts[yName]      # test data-set
          X_tr = dataSet_tr.drop([yName, 'CV'], axis= 1)
          X_ts = dataSet_ts.drop([yName, 'CV'], axis= 1)

          # normalization and standardization
          # NOTE: standardization is applied on train set, 
          # neither the whole data-set nor the test-set
          scaler = StandardScaler().fit(X_tr)
          X_tr_stan = scaler.transform(X_tr)
          X_ts_stan = scaler.transform(X_ts)
          
          # cross-validator
          validator = RepeatedKFold(n_splits= foldCount, n_repeats= 1, random_state= 1)

          ridge_mspes.append(calc_ridge(validator, X_tr_stan, X_ts_stan, y_tr, y_ts))
          lasso_args = calc_lasso(validator, X_tr_stan, X_ts_stan, y_tr, y_ts)
          lasso_mspes.append(lasso_args[0])
          lasso_nfeat.append(lasso_args[1])

          #
          # PCR (Modeling)

     # drawing
     pl.plot(ridge_mspes, 'r')
     pl.plot(lasso_mspes, 'b')
     pl.xlabel('Pricipal Components Count')
     pl.ylabel('MSPE')
     pl.grid()
     pl.show()

except Exception as p:
     print(f'error {inspect.stack()[0][3]}, -> {p.args}')
     pass
