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
	# TODO: based on the len(combos) decide how many figures to plot as there's a max of 9 subplots in mpl
    import numpy as np
    from bokeh.plotting import show

    columns = list(filter(lambda x: x not in exclude_columns, df.columns))
    assert len(columns) > 1, "Too few columns"
    #assert len(columns) < 20, "Too many columns"
    numerical_columns = filter(lambda x: df[x].dtype in [np.float64, np.int64] ,columns)
    combos = list(itertools.combinations(numerical_columns, 2))
    figures, combo_lists = get_figures_and_combos(combos)
    print(figures, combo_lists)
    assert len(figures) == len(combo_lists), "figures not equal to plot groups"
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
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
            combos = itertools.combinations(categories, 2)
            for combo in combos:
                #assert len(categories) == 2, "Only two categories supported at the moment"
                print("# Correlation btw Columns %s & %s by measure %s" % (combo[0],
                                                                                combo[1],
                                                                                meas))
                heatmap = plotter.heatmap(df, combo[0], combo[1],
                                      meas, title="%s vs %s %s heatmap"%(combo[0],
                                                                         combo[1],
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
    if 'stationarity' in kwargs:
        tsu.test_stationarity(ts, valueCol=valueCol,
                                  title=plot_title,
                                  skip_stationarity=skip_stationarity,
                                  **kwargs.get('stationarity'))
    else:
        tsu.test_stationarity(ts, valueCol=valueCol,
                                  title=plot_title,
                                  skip_stationarity=skip_stationarity
                                    )

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

def cluster_analyze(dataframe, cluster_type='KMeans', n_clusters=None):

    # coloured area plots ??)
    from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, SpectralClustering, Birch
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import time

    dataframe = dataframe.as_matrix()
    if cluster_type == 'KMeans':
        assert n_clusters, "Number of clusters argument mandatory"
        cluster_callable = KMeans
        # seed of 10 for reproducibility.
        clusterer = cluster_callable(n_clusters=n_clusters, random_state=10)
    elif cluster_type ==  'dbscan':
        assert not n_clusters, "Number of clusters irrelevant for cluster type : %s"%(cluster_type)
        cluster_callable = DBSCAN
        clusterer = cluster_callable(eps=0.5)
    elif cluster_type == 'affinity_prob':
        assert not n_clusters, "Number of clusters irrelevant for cluster type : %s"%(cluster_type)
        clusterer = AffinityPropagation(damping=.9, preference=-200)
    elif cluster_type == 'spectral':
        assert n_clusters, "Number of clusters argument mandatory"
        clusterer = SpectralClustering(n_clusters=n_clusters,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
    elif cluster_type == 'birch':
        assert not n_clusters, "Number of clusters irrelevant for cluster type : %s"%(cluster_type)
        clusterer = Birch(n_clusters=2)
    else:
        raise "Unknown clustering algorithm type"
    plt.figure(figsize=(2 + 3, 9.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
    t0 = time.time()
    clusterer.fit(dataframe)
    t1 = time.time()
    if hasattr(clusterer, 'labels_'):
        y_pred = clusterer.labels_.astype(np.int)
    else:
        y_pred = clusterer.predict(dataframe)
    print(t0,t1)
    # plot
    plt.subplot(4, 1 , 1)
    if i_dataset == 0:
        plt.title(name, size=18)
    plt.scatter(dataframe[:, 0], dataframe[:, 1], color=colors[y_pred].tolist(), s=10)

    if hasattr(clusterer, 'cluster_centers_'):
        centers = clusterer.cluster_centers_
        center_colors = colors[:len(centers)]
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
				transform=plt.gca().transAxes, size=15,
				horizontalalignment='right')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    plot_num += 1
    plt.show()

def silhouette_analyze(dataframe, cluster_type='KMeans', n_clusters=None):
    # Use clustering algorithms from here
    # http://scikit-learn.org/stable/modules/clustering.html#clustering
    # And add a plot that visually shows the effectiveness of the clusters/clustering rule.(may be
    # coloured area plots ??)

    from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, SpectralClustering, Birch
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    dataframe = dataframe.as_matrix()
    # Silhouette analysis --
    #       http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    #TODO: Add more clustering methods/types like say dbscan and others
    if cluster_type == 'KMeans':
        assert n_clusters, "Number of clusters argument mandatory"
        cluster_callable = KMeans
        # seed of 10 for reproducibility.
        clusterer = cluster_callable(n_clusters=n_clusters, random_state=10)
    elif cluster_type == 'spectral':
        assert n_clusters, "Number of clusters argument mandatory"
        clusterer = SpectralClustering(n_clusters=n_clusters,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
    else:
        raise "Unknown clustering algorithm type"
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.

    #ax1.set_ylim([0, len(dataframe) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    cluster_labels = clusterer.fit_predict(dataframe)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(dataframe, cluster_labels)
    print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dataframe, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(dataframe[:, 0], dataframe[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for %s clustering on sample data "
                    "with n_clusters = %d" % (cluster_type, n_clusters)),
                    fontsize=14, fontweight='bold')

    plt.show()
