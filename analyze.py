import plotter
import itertools

def correlation_analyze(df, exclude_columns = None):
    #TODO: Figure out a way to find correlation between two categorical(ordinal makes more sense) variables(probably by
    #    adding a 3rd, numerical variable??)
    import matplotlib.pyplot as plt
    import numpy as np
    columns = filter(lambda x: x not in exclude_columns, df.columns)
    assert len(df.columns) > 1 and len(df.columns) < 15, "Too many or too few columns"
    numerical_columns = filter(lambda x: df[x].dtype in [np.float64, np.int64] ,columns)
    combos = list(itertools.combinations(numerical_columns, 2))
    # TODO: based on the len(combos) decide how many figures to plot as there's a max of 9 subplots in mpl
    fig = plt.figure(figsize=(20,10))
    for i, combo in enumerate(combos):
        u,v = combo
        # Damn odd way of matplotlib's putting together how many sub plots and which one.
        ax1 = fig.add_subplot(int("3" + str(int(len(combos)/3)) + str(i + 1)))
        plotter.mscatter(ax1, df[u], df[v])
        ax1.set_xlabel(u)
        ax1.set_ylabel(v)
        ax1.legend(loc='upper left')
    plt.show()
    print(df.corr())

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

def time_series_analysis(df):
    import timeseriesUtils as tsu
    # TODO;
    # 1. Do, ADF(Dickey-Fuller's ) stationarity test
    # 2. Seasonal decomposition of the time series and plot it
    # 3. ARIMA model of the times
    # 4. And other time-serie models like AR, etc..
    tsu.test_stationarity()
    tsu.plot_autocorrelation() # AR model
    tsu.seasonal_decompose()
    pass
