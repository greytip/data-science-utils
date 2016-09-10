import pandas as pd
import matplotlib.pyplot as plt

def test_stationarity(timeseries, valueCol, title='timeseries', **kwargs):

    from statsmodels.tsa.stattools import adfuller
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation of ' + title )
    plt.show()

    #Perform Dickey-Fuller test:
    dftest = adfuller(timeseries[valueCol], autolag=kwargs.get('autolag', 't-stat'))
    print('Results of Dickey-Fuller Test:')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def plot_autocorrelation(timeseries_df, valueCol=None,
                         timeCol='timestamp', timeInterval='30min', partial=False):
    """
    Plot autocorrelation of the given dataframe based on statsmodels.tsa.stattools.acf
			(which apparently is simple Ljung-Box model)
    Assumes:
       default timecol == 'timestamp' if different pass a kw parameter

    """
    import statsmodels.api as sm
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    if partial:
        subplt = sm.graphics.tsa.plot_acf(timeseries_df[valueCol].squeeze(), lags=40, ax=ax1)
    else:
        subplt = sm.graphics.tsa.plot_pacf(timeseries_df[valueCol], lags=40, ax=ax1)
    plt.show()
    return fig

def seasonal_decompose(timeseries_df, freq='30min', **kwargs):
    import statsmodels.api as sm
    timeseries_df.interpolate(inplace=True)
    seasonal_components = sm.tsa.seasonal_decompose(timeseries_df, **kwargs)
    fig = seasonal_components.plot()
    return fig

def create_timeseries_df(dataframe, dropColumns=list(),filterByCol=None,
                      filterByVal=None, timeCol='date',
                      timeInterval='30min', func=sum):
    """
    # A simple function that takes df, and returns a timeseries with a temporal distribution of audit events
    auditcode= <specify which audit event> (None means just a distribution of any audit event)

    """
    new_df = dataframe.copy(deep=True)
    new_df.drop(dropColumns, 1, inplace=True)
    if filterByVal:
        assert filterByCol, "Column to be filtered by is mandatory"
        assert filterByCol not in dropColumns, "Cannot group by a column that's to be dropped"
        assert type(filterByCol) != list
        new_df = new_df[new_df[filterByCol].isin(filterByVal)].groupby(timeCol).agg(func)
        new_df.columns = filterByVal
        new_df.index = pd.to_datetime(new_df.index)
        new_df = new_df.resample(timeInterval, func)
    else:
        new_df = new_df.groupby(timeCol).agg(func)
        new_df.index = pd.to_datetime(new_df.index)
        new_df = new_df.resample(timeInterval, func)
    return new_df
