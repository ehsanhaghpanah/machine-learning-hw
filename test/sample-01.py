# %%

import pandas as pa
import os
import inspect

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
     path: str = os.getcwd() + '\\data\\bitcoin-dataset.xlsx'
     dataSet: pa.DataFrame = pa.read_excel(path)

     # {Date} field must be removed because of {StandardScaler}
     X = dataSet.drop(['Date'], axis= 1)

     # defining dependent variable and independent variables (vector)
     yName = 'Bitcoin Market Price USD' 
     y = X[yName]
     # X is a vector so it is in capital
     X = X.drop([yName], axis= 1)

     # splitting (or folding !?) into {traing set} and {validation set}
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state= 1)

     # normalization and standardization
     # NOTE: standardization is applied on train set, 
     # neither the whole data-set nor the test-set
     scaler = StandardScaler().fit(X_train)
     X_train_stan = scaler.transform(X_train)
     X_test_stan  = scaler.transform(X_test)

     # # output
     # print(f'X Train Set Standard = {X_train_stan}')
     # print(f'X Test  Set Standard = {X_test_stan}')

except Exception as p:
     print(f'error {inspect.stack()[0][3]}, -> {p.args}')
     pass

# %%

import matplotlib.pyplot as pl

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

try:

     # Principal Component Analysis
     analysis = PCA()
     # Principal Component(s), we apply analysis fitting on train set not the test set
     pcs_train = analysis.fit_transform(X_train_stan)
     # Principal Component(s),
     pcs_test = analysis.transform(X_test_stan)

     # modeling
     model = LinearRegression()
     MSPEs = []

     featureCounts: int = X.shape[1] + 1
     for featureIndex in range(1, featureCounts):
          # creating a model over (xTrain, yTrain)
          model.fit(pcs_train[:, : featureIndex], y_train)
          # predicting yTest(Predicted) based on xTest(DataSet)
          guess = model.predict(pcs_test[:, : featureIndex])
          # calculating residuals of yTest(Predicted) and yTest(DataSet)
          error = mean_squared_error(y_test, guess)
          MSPEs.append(error)

     # # output
     # print(MSPEs)
     
     # drawing
     pl.plot(MSPEs)
     pl.xlabel('Pricipal Components Count')
     pl.xlabel('MSPE')
     pl.grid()
     pl.show()

except Exception as p:
     print(f'error {inspect.stack()[0][3]}, -> {p.args}')
     pass

# %%

try:

     selectedFeaturesCount: int = 15
     model.fit(pcs_train[:, : selectedFeaturesCount], y_train)
     guess = model.predict(pcs_test[:, : selectedFeaturesCount])

     print(f'MSPE = {mean_squared_error(y_test, guess)}')
     print(f'beta0 = {model.intercept_}')
     print(f'beta* = {model.coef_}')

except Exception as p:
     print(f'error {inspect.stack()[0][3]}, -> {p.args}')
     pass

# %%

import numpy as np

from IPython.display import display

# NOTE: this type of cross-validation may not be suitable for 
# this data-set as the original data-set is a time-series
totalRecords = dataSet.shape[0]
foldCount = 10
crossValidation = np.random.randint(low= 0, high= foldCount, size= totalRecords)
dataSet['CV'] = crossValidation

# display(dataSet)
print(dataSet['CV'].value_counts())
print(dataSet.isnull().sum())
