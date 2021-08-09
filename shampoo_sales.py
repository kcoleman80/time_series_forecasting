#!/usr/bin/env python
# coding: utf-8

# In[9]:


from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas.plotting import lag_plot
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error


# In[10]:


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv'
series = read_csv(url, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))


# In[11]:


lag_plot(series)
pyplot.show()


# In[12]:


#correlation
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
pyplot.show()


# In[14]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=31)
pyplot.show()


# In[15]:


# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]


# In[16]:


# persistence model
def model_persistence(x):
    return x


# In[17]:


# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)


# In[18]:


# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()


# In[19]:


# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()


# # Autoregression

# In[20]:


# create and evaluate a static autoregressive model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[21]:


# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv'
series = read_csv(url, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)


# In[22]:


# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]


# In[24]:


# train autoregression
model = AutoReg(train, lags=10)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)


# In[25]:


# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# In[26]:


# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# An alternative would be to use the learned coefficients and manually make predictions. This requires that the history of 29 prior observations be kept and that the coefficients be retrieved from the model and used in the regression equation to come up with new forecasts.

# In[27]:


# create and evaluate an updated autoregressive model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[28]:


# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv'
series = read_csv(url, header=0, index_col=0, parse_dates=True, squeeze=True)


# In[29]:


# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]


# In[31]:


# train autoregression
window = 10
model = AutoReg(train, lags=10)
model_fit = model.fit()
coef = model_fit.params


# In[32]:


# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# In[33]:


# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# # Arima 

# In[34]:


from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


# In[35]:


# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv'
series = read_csv(url, header=0, index_col=0, parse_dates=True, squeeze=True,date_parser=parser)
print(series.head())


# In[36]:


series.plot()
pyplot.show()


# In[37]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
pyplot.show()


# In[38]:


from statsmodels.tsa.arima.model import ARIMA


# In[39]:


series.index = series.index.to_period('M')


# In[40]:


# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()


# In[41]:


# summary of fit model
print(model_fit.summary())


# In[42]:


# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()


# In[43]:


# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())


# # Rolling Forecast ARIMA Model

# In[44]:


# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[45]:


# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
# Create lagged dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv'
series = read_csv(url, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')


# In[46]:


# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()


# In[47]:


# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


# In[48]:


# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# In[49]:


# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[ ]:





# In[ ]:




