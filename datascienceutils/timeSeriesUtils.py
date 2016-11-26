import pandas as pd
from bokeh.plotting import figure, show

from . import plotter
def test_stationarity(timeseries, timeCol, valueCol, skip_stationarity=False, title='timeseries', **kwargs):

    from statsmodels.tsa.stattools import adfuller
    calcStatsDf = pd.DataFrame()
    #Determing rolling statistics
    calcStatsDf['rollingMean'] = pd.rolling_mean(timeseries, window=12)[valueCol]
    calcStatsDf['rollingSTD']  = pd.rolling_std(timeseries, window=12)[valueCol]
    timeseries = timeseries.reset_index()
    calcStatsDf['time'] = timeseries[timeCol]

    #Plot rolling statistics:
    fig = figure(width=12, height=8)
    orig = plotter.lineplot(timeseries, timeCol, valueCol, color='blue',label='Original')
    mean = plotter.lineplot(calcStatsDf,'time', 'rollingMean', fig=fig, color='red', label='Rolling Mean')
    std = plotter.lineplot(calcStatsDf, 'time', 'rollingSTD', fig=fig, color='black', label = 'Rolling Std')

    if not skip_stationarity:
        #Perform Dickey-Fuller test:
        dftest = adfuller(timeseries[valueCol], autolag=kwargs.get('autolag', 't-stat'))
        print('Results of Dickey-Fuller Test:')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
    return fig

def plot_autocorrelation(timeseries_df, valueCol=None,
                         timeCol='timestamp', timeInterval='30min', partial=False):
    """
    Plot autocorrelation of the given dataframe based on statsmodels.tsa.stattools.acf
			(which apparently is simple Ljung-Box model)
    Assumes:
       default timecol == 'timestamp' if different pass a kw parameter

    """
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    if not partial:
        plt = sm.graphics.tsa.plot_acf(timeseries_df[valueCol].squeeze(), lags=40, ax=ax1)
    else:
        plt = sm.graphics.tsa.plot_pacf(timeseries_df[valueCol], lags=40, ax=ax1)
    plt.show()
    return plt

def seasonal_decompose(timeseries_df, freq=None, **kwargs):
    import statsmodels.api as sm
    timeseries_df.interpolate(inplace=True)
    print(freq)
    if not freq: freq = len(timeseries_df) - 2
    seasonal_components = sm.tsa.seasonal_decompose(timeseries_df, freq=freq, **kwargs)
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
    if dropColumns:
        new_df.drop(dropColumns, 1, inplace=True)
    if filterByVal:
        assert type(filterByVal) == list, "Need a list of values for filterByVal"
        assert filterByCol, "Column to be filtered by is mandatory"
        assert filterByCol not in dropColumns, "Cannot group by a column that's to be dropped"
        assert type(filterByCol) != list, "Only single column can be passed"
        new_df = new_df[new_df[filterByCol].isin(filterByVal)].groupby(timeCol).agg(func)
        new_df.columns = filterByVal
        new_df.index = pd.to_datetime(new_df.index)
        new_df = new_df.resample(timeInterval, func)
    else:
        new_df = new_df.groupby(timeCol).agg(func)
        new_df.index = pd.to_datetime(new_df.index)
        new_df = new_df.resample(timeInterval, func)
    return new_df
