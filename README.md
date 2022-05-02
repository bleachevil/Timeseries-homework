# Timeseries-homework

## Time series analysis
## Return Forecasting: Time Series Analysis & Modelling with CAD-PHY Exchange rate data (input and clear data)
### import model
```
import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=Warning)
```

### import data
```
cad_jpy_df = pd.read_csv(
    Path("cad_jpy.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
cad_jpy_df.head()
cad_jpy_df = cad_jpy_df.loc["1990-01-01":, :]
cad_jpy_df.head()
```

### Initial Time-Series Plotting
```
cad_jpy_df.Price.plot(figsize=(20,10))
```
### result
picture

### Question and answer
Question: Do you see any patterns, long-term and/or short? <br />
Answer: the trend went down from long term and short term perspective. the support line was around 70.

### Decomposition Using a Hodrick-Prescott Filter

```
import statsmodels.api as sm
noise, trend = sm.tsa.filters.hpfilter(cad_jpy_df['Price'])
df_noise = pd.DataFrame(noise)
df_noise.rename(columns = {'Price_cycle':'noise'},inplace=True)
df_trend = pd.DataFrame(trend)
df_trend.rename(columns = {'Price_trend':'trend'},inplace=True)
df_price = cad_jpy_df[['Price']]
df = pd.concat([df_price,df_noise,df_trend], axis= 'columns')
df
```
### result and plot
picture1&2

### Question and answer
Question: Do you see any patterns, long-term and/or short? <br />
Answer: the trend is matching with the price. 

### plot noise
```
df['noise'].plot(figsize=(20,10),title='Noise')
```
picture 

### Forecasting Returns using an ARMA Model
```
returns = (cad_jpy_df[["Price"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.tail()
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
model = ARMA(returns.values,order=(2,1))
results = model.fit()
results.summary()
```

### result of summary for ARMA
picture arma 1

### Plot the 5 Day Returns Forecast
```
res = results.forecast(steps=5)
df_res = pd.DataFrame(res[0])
df_res.plot()
```
picture arma 2 <br />

### Question and answer
Question: Based on the p-value, is the model a good fit? <br />
Answer: the model is good fit.

### Forecasting the Exchange Rate Price using an ARIMA Model
```
from statsmodels.tsa.arima_model import ARIMA
model_1 = ARIMA(cad_jpy_df["Price"], order=(5, 1, 1))
results_ARIMA = model_1.fit()
results_ARIMA.summary()
```

### result of summary for ARIMA
Picture of ARIMA

### Plot the 5 Day Price Forecast
```
res = results_ARIMA.forecast(steps=5)
df_res = pd.DataFrame(res[0])
df_res.plot(figsize=(20,10))
```
picture of ARIMA2

### Question and answer
Question: What does the model forecast will happen to the Japanese Yen in the near term? <br />
Answer: the price will go down.

### Volatility Forecasting with GARCH
```
from arch import arch_model
returns = cad_jpy_df[["Price"]].pct_change() * 100
returns = returns.replace(-np.inf, np.nan).dropna()
model_arch = arch_model(returns['Price'], mean="Zero", vol="GARCH", p=1, q=1)
res = model_arch.fit(disp="off")
res.summary()
```

### result of summary for GARCH model
picture of GARCH


### volatility forecasts
```
last_day = returns.index.max().strftime('%Y-%m-%d')
last_day
forecast_horizon = 5
forecasts = res.forecast(start='2020-06-04', horizon=forecast_horizon)
intermediate = np.sqrt(forecasts.variance.dropna() * 252)
intermediate.head()
final = intermediate.dropna().T
final.head()
final.plot()
```
### forecast reuslt
picture of GARCH 2

### Question and answer
Question: What does the model forecast will happen to volatility in the near term? <br />
Answer: the price of Yan is very volatile.

### Conculsions
1) Based on your time series analysis, would you buy the yen now?<br />
i will not buy the yen now.<br />

2) Is the risk of the yen expected to increase or decrease?<br />
the risk of the yen expected to increase.<br />

3) Based on the model evaluation, would you feel confident in using these models for trading?<br />
maybe i think i need to do more testing to make the conculsion.<br />


## Regression analysis
## Regression Analysis: Seasonal Effects with Sklearn Linear Regression

### import model
```
import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline
```

### import data
```
cad_jpy_df = pd.read_csv(
    Path("cad_jpy.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True)
cad_jpy_df.head()
cad_jpy_df = cad_jpy_df.loc["1990-01-01":, :]
cad_jpy_df.head()
```
### data preparation
```
cad_jpy_df['Return'] = cad_jpy_df.Price.pct_change()
cad_jpy_df.dropna(inplace=True)
cad_jpy_df.head()
```
picture of data prepartion
### Lagged Returns
```
cad_jpy_df['Lagged_Return'] = cad_jpy_df.Return.shift()
cad_jpy_df.dropna(inplace=True)
cad_jpy_df.head()
```
picture of lagged returns

### Train Test Split
```
train = cad_jpy_df[:'2017']
test = cad_jpy_df['2018':]
X_train = train["Lagged_Return"].to_frame()
X_test = test["Lagged_Return"].to_frame()
y_train = train["Return"]
y_test = test["Return"]
X_train.head()
```
picture of data x train

### Linear Regression Model
```
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### Make predictions using the Testing Data
```
predictions = model.predict(X_test)
Results = y_test.to_frame()
Results["Predicted Return"] = predictions
Results.head(2)
Results[:20].plot(subplots=True)
```
picture of prediction

### Out-of-Sample Performance
```
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(
    Results["Return"],
    Results["Predicted Return"]
)
rmse = np.sqrt(mse)
print(f"Out-of-Sample Root Mean Squared Error (RMSE): {rmse}")
```
#### result
Out-of-Sample Root Mean Squared Error (RMSE): 0.006445805658569028

### In-Sample Performance
```
# Construct a dataframe using just the "y" training data:
in_sample_results = y_train.to_frame()

# Add a column of "in-sample" predictions to that dataframe:  
in_sample_results["In-sample Predictions"] = model.predict(X_train)

# Calculate in-sample mean_squared_error (for comparison to out-of-sample)
in_sample_mse = mean_squared_error(
    in_sample_results["Return"],
    in_sample_results["In-sample Predictions"]
)

# Calculate in-sample root mean_squared_error (for comparison to out-of-sample)
in_sample_rmse = np.sqrt(in_sample_mse)
print(f"In-sample Root Mean Squared Error (RMSE): {in_sample_rmse}")
```
#### result
In-sample Root Mean Squared Error (RMSE): 0.00841994632894117

### Conclusions
Question: Does this model perform better or worse on out-of-sample data as compared to in-sample data?<br />

Answer: the model perform worse on out-of-sample data as compared to in-sample data.
