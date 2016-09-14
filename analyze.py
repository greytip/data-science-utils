import plotter
import itertools
import matplotlib.pyplot as plt

def chunks(combos, size=9):
    for i in range(0, len(combos), size):
        yield combos[i:i + size]

def get_figures_and_combos(combos):
    figures = list()
    combo_lists = list()
    for each in chunks(combos, 9):
        figures.append(plt.figure(figsize=(20,10)))
        combo_lists.append(each)
    return figures, combo_lists

def correlation_analyze(df, exclude_columns = None, categories=[], measure=None):
    import numpy as np
    from bokeh.plotting import show

    columns = list(filter(lambda x: x not in exclude_columns, df.columns))
    assert len(columns) > 1, "Too few columns"
    #assert len(columns) < 20, "Too many columns"
    numerical_columns = filter(lambda x: df[x].dtype in [np.float64, np.int64] ,columns)
    combos = list(itertools.combinations(numerical_columns, 2))
    figures, combo_lists = get_figures_and_combos(combos)
    print(figures, combo_lists)
    assert len(figures) == len(combo_lists), "figures not equal to plot groups "
    # TODO: based on the len(combos) decide how many figures to plot as there's a max of 9 subplots in mpl
    for i, figure in enumerate(figures):
        for combo in combo_lists[i]:
            u,v = combo
            # Damn odd way of matplotlib's putting together how many sub plots and which one.
            ax1 = figure.add_subplot(int("3" + str(int(len(combo_lists[i])/3)) + str(i + 1)))
            plotter.mscatter(ax1, df[u], df[v])
            ax1.set_xlabel(u)
            ax1.set_ylabel(v)
            #ax1.legend(loc='upper left')

    print("# Correlation btw Numerical Columns")
    plt.show()
    if (categories and measure):
        #TODO; Add support for more categorical variables
        for meas in measure:
            assert len(categories) == 2, "Only two categories supported at the moment"
            print("# Correlation btw Columns %s & %s by measure %s" % (categories[0],
                                                                                categories[1],
                                                                                meas))
            heatmap = plotter.heatmap(df, categories[0], categories[1],
                                      meas, title="%s vs %s %s heatmap"%(categories[0],
                                                                         categories[1],
                                                                         meas))
            show(heatmap)
    print("# Pandas correlation coefficients matrix")
    print(df.corr())
    # Add co-variance matrix http://scikit-learn.org/stable/modules/covariance.html#covariance
    print("# Pandas co-variance coefficients matrix")
    print(df.cov())

def regression_analyze(df, col1, col2, trainsize=0.8):
    """
    """
    import sklearnUtils as sku

    from sklearn.cross_validation import cross_val_predict, train_test_split, cross_val_score
    import matplotlib.pyplot as plt
    import numpy as np

    # this is the quantitative/hard version of teh above
    #TODO: Simple  line plots of column variables, but for the y-axis, # Fit on
    #         a, linear function(aka line)
    #         b, logarithmic/exponetial function
    #         c, logistic function
    #         d, parabolic function??
    #   Additionally plot the fitted y and the correct y in different colours against the same x
    new_df = df[[col1, col2]].copy(deep=True)
    target = new_df[col2]
    models = [  sku.train(new_df, target, col1, modelType='linearRegression'),
                sku.train(new_df, target, col1, modelType='logisticRegression'),
              ]
    map(fit, models)
    pass

def time_series_analysis(df, timeCol='date', valueCol=None, timeInterval='30min',
                         plot_title = 'timeseries',
                         skip_stationarity=False,
                         skip_autocorrelation=False,
                         skip_seasonal_decompose=False, **kwargs):
    import timeSeriesUtils as tsu
    if 'create' in kwargs:
        ts = tsu.create_timeseries_df(df, timeCol=timeCol, timeInterval=timeInterval, **kwargs.get('create'))
    else:
        ts = tsu.create_timeseries_df(df, timeCol=timeCol, timeInterval=timeInterval)
    # TODO;
    # 1. Do, ADF(Dickey-Fuller's ) stationarity test
    # 2. Seasonal decomposition of the time series and plot it
    # 3. ARIMA model of the times
    # 4. And other time-serie models like AR, etc..
    if not skip_stationarity:
        if 'stationarity' in kwargs:
            tsu.test_stationarity(ts, valueCol=valueCol, title=plot_title,
                                                    **kwargs.get('stationarity'))
        else:
            tsu.test_stationarity(ts, valueCol=valueCol, title=plot_title)

    if not skip_autocorrelation:
        if 'autocorrelation' in kwargs:
            tsu.plot_autocorrelation(ts, valueCol=valueCol, **kwargs.get('autocorrelation')) # AR model
            tsu.plot_autocorrelation(ts, valueCol=valueCol, partial=True, **kwargs.get('autocorrelation')) # partial AR model
        else:
            tsu.plot_autocorrelation(ts, valueCol=valueCol) # AR model
            tsu.plot_autocorrelation(ts, valueCol=valueCol, partial=True) # partial AR model

    if not skip_seasonal_decompose:
        if 'seasonal' in kwargs:
            seasonal_args = kwargs.get('seasonal')

            tsu.seasonal_decompose(ts, **seasonal_args)
        else:
            tsu.seasonal_decompose(ts)

def cluster_analyze():
    from sklearn.cluster import KMeans
    # Use clustering algorithms from here
    # http://scikit-learn.org/stable/modules/clustering.html#clustering
    # And add a plot that visually shows the effectiveness of the clusters/clustering rule.(may be
    # coloured area plots ??)
    pass
