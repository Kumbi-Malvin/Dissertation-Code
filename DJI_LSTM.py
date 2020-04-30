import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2


# Importing the training set
dataset_train = pd.read_csv('DJI-Train.csv')
training_set = dataset_train.iloc[:, 5:6].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating Training set so that 60 inputs produce the next input as explained in the dissertation. 
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Obtaining Test data for validation
dataset_test = pd.read_csv('DJI-Test.csv')
real_stock_price = dataset_test.iloc[:, 5:6].values


# Using the Test data for validation
total_dataset = pd.concat((dataset_train['Adj Close'], dataset_test['Adj Close']), axis = 0)
inputs = total_dataset[len(total_dataset) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.array(real_stock_price)
y_test = sc.transform(y_test)

# Constructing the LSTM Network architectures

# Initialising the RNN
regressor = Sequential()

# The LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 300, input_shape = (X_train.shape[1], 1))) # Manipulate units for the number of LSTM units
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 50,validation_data=(X_test, y_test))

# Forecasting Process

pred = []

batch = X_train[-1].reshape((1,60,1))

for i in range(len(real_stock_price)+1):
    pred.append(regressor.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[pred[i]]],axis = 1)
pred_list = sc.inverse_transform(pred)

# Visualising the results

plt.plot(real_stock_price, color = 'red', label = 'Real Dow Jones Price')
plt.plot(pred_list, color = 'green', label = 'Forecasted Dow Jones Price')
plt.title('Dow Jones Price Prediction and Forecast')
plt.xlabel('Time')
plt.ylabel('Dow Jones Price')
plt.legend()
plt.show()

# Loss as a function of the number of epochs graph

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Forecast Performance indicatiors

forecast_mse = mse(real_stock_price,pred_list[1:])
forecast_rmse = mse(real_stock_price,pred_list[1:], squared = False)
forecast_r2 = r2(real_stock_price,pred_list[1:])
print('Forecast MSE: ',forecast_mse)
print('Forecast RMSE: ',forecast_rmse)
print('Forecast R^2: ',forecast_r2)