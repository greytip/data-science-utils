import plotter
import itertools

def correlation_analyze(df):
    #TODO: Plot scatter plot of all combinations of column variables in the df
    import matplotlib.pyplot as plt
    import numpy as np
    columns = df.columns
    assert len(columns) > 1 and len(columns) < 15, "Too many or too few columns"
    numerical_columns = filter(lambda x: df[x].dtype in [np.float64, np.int64] ,columns)
    combos = list(itertools.combinations(numerical_columns, 2))
    fig = plt.figure()
    for i, combo in enumerate(combos):
        u,v = combo
        # Damn odd way of matplotlib's putting together how many sub plots and which one.
        ax1 = fig.add_subplot(int("3" + str(int(len(combos)/3)) + str(i + 1)))
        plotter.mscatter(ax1, df[u], df[v])
    plt.legend(loc='upper left')
    plt.show()
    print(df.corr())

def regression_analyze(df, col1, col2):
    """
    """
    # this is the quantitative/hard version of teh above
    #TODO: Simple  line plots of column variables, but for the y-axis, # Fit on
    #         a, linear function(aka line)
    #         b, logarithmic/exponetial function
    #         c, logistic function
    #         d, parabolic function??
    #   Additionally plot the fitted y and the correct y in different colours against the same x
    pass

def time_series_analysis(df):
    # TODO;
    # 1. Do, ADF(Dickey-Fuller's ) stationarity test
    # 2. Seasonal decomposition of the time series and plot it
    # 3. ARIMA model of the times
    # 4. And other time-serie models like AR, etc..
    pass
