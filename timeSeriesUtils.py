def test_stationarity(timeseries):

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
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

def plot_autocorrelation(dataframe, timeCol='timestamp', timeInterval='30min', partial=False):
    """
    Plot autocorrelation of the given dataframe based on statsmodels.tsa.stattools.acf
			(which apparently is simple Ljung-Box model)
    Assumes:
       default timecol == 'timestamp' if different pass a kw parameter

    """
	import statsmodels.api as sm
    timeseries_df = create_timeseries(dataframe, timeInterval=timeInterval)
	fig = plt.figure(figsize=(12,8))
	ax1 = fig.add_subplot(1)
	if partial:
		fig = sm.graphics.tsa.plot_acf(audit_events.values.squeeze(), lags=40, ax=ax1)
	else:
		fig = sm.graphics.tsa.plot_pacf(audit_events, lags=40,ax=ax2)
	return fig

def seasonal_decompose(dataframe, timeCol='timestamp', timeInterval='30min'):
	"""
	"""
	import statsmodels.api as sm
    timeseries_df = create_timeseries(dataframe, timeCol=timeCol, timeInterval=timeInterval)
	timeseries_df.interpolate(inplace=True)
	seasonal_components = sm.tsa.seasonal_decompose(audit_events.values, model='additive', freq=24)
	fig = seasonal_components.plot()
	return fig

def create_timeseries(dataframe, dropColumns=list(), groupByCol=list(),
                      filterByCol=None, filterByVal=list(), timeCol='auditdate',
                      timeInterval='30min', func=sum):
    """
    # A simple function that takes the audit_df, and returns a df with a temporal distribution of audit events
    auditcode= <specify which audit event> (None means just a distribution of any audit event)

    """
    assert groupByCol not in dropColumns, "Cannot group by a column that's to be dropped"
    assert type(filterByCol) != list
    new_df = dataframe.drop(dropColumns, 1)
    if filterByVal:
        if groupByCol:
            new_df = new_df[new_df[filterByCol].isin(filterByVal)].groupby(groupByCol + timeCol).count()
            new_df.columns = filterByVal
            new_df = new_df[filterByVal].unstack(groupByCol).resample(timeInterval, func).stack(groupByCol)
        else:
            new_df = new_df[new_df[filterByCol].isin(filterByVal)].groupby(timeCol).count()
            new_df.columns = filterByVal
            new_df = new_df.resample(timeInterval, func)
    else:
        if groupByCol:
            new_df = new_df.groupby(groupByCol + timeCol).count()
            new_df.columns = filterByVal
            new_df = new_df[filterByVal].unstack(groupByCol).resample(timeInterval, func).stack(groupByCol)
        else:
            new_df = new_df.groupby(timeCol).count()
            new_df = new_df.resample(timeInterval, func)
    return new_df
