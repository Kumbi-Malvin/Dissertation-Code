import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from arch.unitroot import ADF

def parser(x):
    return datetime.strptime(x,'%d/%m/%Y')

data = pd.read_csv('DJI-Train_ARIMA.csv',index_col=0,parse_dates= [0],date_parser = parser)
test_data = pd.read_csv('DJI-Test_ARIMA.csv',index_col=0,parse_dates = [0],date_parser = parser)

# Augmented Dickey Fuller Test for Stationarity

adf_test = ADF(data,regression = 'c')
adf_test.lags = 2
adf.trend = 'c'
print(adf_test.summary().as_text())

plt.plot(data)

# ACF and PACF of Raw data
plot_acf(data)
plot_pacf(data)

### Stationarity Conversion ###

data_diff = data.diff(periods = 1)
data_diff = data_diff[1:]

### ACF and PACF for differenced Time Series ###
plot_acf(data_diff)
plot_pacf(data_diff)

### Stationarity Conversion ###

second_diff = data.diff(periods = 2)
second_diff = second_diff[2:]

### ACF and PACF for second differenced Time Series ###
plot_acf(second_diff)
plot_pacf(second_diff)

# ARMA Model Selection (Second Differencing) #
for i in range(9):
    train = data.values
    test = test_data.values
    model_arima = ARIMA(train,order=(9-i,2,2))
    model_arima_fit = model_arima.fit()
    print(model_arima_fit.summary())
    print('#############################################################################\n')

chosen_model = ARIMA(train,order=(1,2,2))
chosen_model_fit = chosen_model.fit()
forecast = chosen_model_fit.forecast(steps = len(test))
rmse_forc = mse(forecast[0],test,squared = False)
print('RMSE: ',rmse_forc)