import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


a = pd.read_csv('covidNewCases.csv')
x = pd.read_csv('covidNewDs.csv')
# Load the diabetes dataset
#diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
a = a[:, np.newaxis, 2]

# Split the data into training/testing sets
a_X_train = a[:-20]
a_X_test = a[-20:]

# Split the targets into training/testing sets
x_y_train = x[:-20]
x_y_test = x[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(a_X_train, x_y_train)

# Make predictions using the testing set
x_y_pred = regr.predict(a_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(x_y_test, x_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(x_y_test, x_y_pred))

# Plot outputs
plt.scatter(a_X_test, x_y_test,  color='black')
plt.plot(a_X_test, x_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()