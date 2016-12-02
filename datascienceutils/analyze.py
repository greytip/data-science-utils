# Standard and external libraries
from bokeh.plotting import show
from bokeh.layouts import gridplot

import itertools
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Custom libraries
from . import plotter
from . import utils

def dist_analyze(df, column=None, categories=[]):
    if not column:
        plots=[]
        numericalColumns = df.select_dtypes(include=[np.number]).columns
        for column in numericalColumns:
            plots.append(plotter.sb_violinplot(df[column], inner='box'))
        catColumns = set(df.columns).difference(set(numericalColumns))
        for column in catColumns:
            plots.append(plotter.pieChart(df, column))
        grid = gridplot(list(utils.chunks(plots, size=2)))
        show(grid)
        if categories:
            # Plot Barplots of combination of category and numerical columns
            catNumCombos = set(itertools.product(numericalColumns, categories))
            barplots = []
            for each in catNumCombos:
                barplots.append(plotter.barplot(df, each[1], each[0]))
            print("# Joint Distribution of Numerical vs Categorical Columns")
            grid = gridplot(list(utils.chunks(barplots, size=2)))
            show(grid)
    else:
        show(plotter.sb_violinplot(df[column], inner='box'))

def correlation_analyze(df, exclude_columns = [], categories=[], measures=None, trellis=False):
    """
    Plot scatter plots of all combinations of numerical columns.
    If categorise and measures are passed, plot heatmap of combination of categories by measure.
    @params:
        df: Dataframe table data.
        exclude_columns: Columns to be excluded/ignored
        categories: list of categorical variable names
        measures: List of measures to plot heatmap of categories
        trellis: Plot trellis type plots for the categories only valid if categories is passed
                (inspired by this:
                http://github.com/anandjeyahar/statistical-analysis-python-tutorial/3.%20Plotting%20and%20Visualization.ipynb)
    """
    columns = set(filter(lambda x: x not in exclude_columns, df.columns))
    assert len(columns) > 1, "Too few columns"
    if not measures:
        measures = ['count']

    # Plot scatter plot of combination of numerical columns
    numericalColumns = set(df.select_dtypes(include=[np.number]).columns).intersection(columns)
    combos = list(itertools.combinations(numericalColumns, 2))
    plots = []
    for combo in combos:
        u,v = combo
        plots.append(plotter.scatterplot(df, u, v))

    print("# Correlation btw Numerical Columns")
    grid = gridplot(list(utils.chunks(plots, size=2)))
    show(grid)

    if (categories and measures):
        # Plot heatmaps of category-combos by measure value.
        heatmaps = []
        combos = itertools.combinations(categories, 2)
        cols = list(df.columns)
        if 'count' in measures:
            # Do a group by on categories and use count() to heatmap
            measures.remove('count')
            for combo in combos:
                counts = list()
                print("# Correlation btw Columns %s & %s by count" % (combo[0], combo[1]))
                group0 = df.groupby(list(combo)).size()
                for idx, each in df.iterrows():
                    counts.append(each[cols.index(combo[0])][cols.index(combo[1])])
                df['counts'] = counts
                heatmaps.append(plotter.heatmap(df, combo[0], combo[1], 'counts'))
            df.drop('counts', 1, inplace=True)

        for meas in measures:
            # Plot heatmaps for measure across all combination of categories
            for combo in combos:
                print("# Correlation btw Columns %s & %s by measure %s" % (combo[0],
                                                                                combo[1],
                                                                                meas))
                heatmaps.append(plotter.heatmap(df, combo[0], combo[1],
                                      meas, title="%s vs %s %s heatmap"%(combo[0],
                                                                         combo[1],
                                                                         meas)))
        hmGrid = gridplot(list(utils.chunks(heatmaps, size=2)))
        show(hmGrid)
        if trellis:
            trellisPlots = list()
    print("# Pandas correlation coefficients matrix")
    print(df.corr())
    # Add co-variance matrix http://scikit-learn.org/stable/modules/covariance.html#covariance
    print("# Pandas co-variance coefficients matrix")
    print(df.cov())

def regression_analyze(df, col1, col2, trainsize=0.8):
    """
    Plot regressed data vs original data for the passed columns.
    """
    from . import predictiveModels as pm

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
    models = [  pm.train(new_df, target, column=col1, modelType='linearRegression'),
                #pm.train(new_df, target, column=col1, modelType='logisticRegression'),
              ]
    plots = list()
    for model in models:
        scatter = plotter.scatterplot(new_df, col1, col2)
        source = new_df[col1].as_matrix().reshape(-1,1)
        flatSrc = [item for sublist in source for item in sublist]
        predicted = list(model.predict(source))
        scatter.line(flatSrc,
                     predicted,
                     line_color='red')
        plots.append(scatter)
        show(scatter)
        print("Regression Score")
        print(model.score(source, new_df[col2].as_matrix().reshape(-1,1)))

    pass

def time_series_analysis(df, timeCol='date', valueCol=None, timeInterval='30min',
                         plot_title = 'timeseries',
                         skip_stationarity=False,
                         skip_autocorrelation=False,
                         skip_seasonal_decompose=False, **kwargs):
    """
    Plot time series, rolling mean, rolling std , autocorrelation plot, partial autocorrelation plot
    and seasonal decompose
    """
    from . import timeSeriesUtils as tsu
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
        plt = tsu.test_stationarity(ts, timeCol=timeCol, valueCol=valueCol,
                                  title=plot_title,
                                  skip_stationarity=skip_stationarity,
                                  **kwargs.get('stationarity'))
    else:
        plt = tsu.test_stationarity(ts, timeCol=timeCol, valueCol=valueCol,
                                  title=plot_title,
                                  skip_stationarity=skip_stationarity
                                    )
    show(plt)
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
    """
    Apply the given clustering method and plot scatter plot and center
    """

    # coloured area plots ??)
    from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, SpectralClustering, Birch
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import time

    df_mat = dataframe.as_matrix()
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
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    #plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
    t0 = time.time()
    clusterer.fit(df_mat)
    t1 = time.time()
    if hasattr(clusterer, 'labels_'):
        y_pred = clusterer.labels_.astype(np.int)
    else:
        y_pred = clusterer.predict(df_mat)
    dataframe['y_pred'] = y_pred
    # plot
    plt.title(cluster_type, size=18)
    plt.scatter(df_mat[:, 0], df_mat[:, 1]) # color=colors[y_pred].tolist(), s=10)

    if hasattr(clusterer, 'cluster_centers_'):
        centers = clusterer.cluster_centers_
        center_colors = colors[:len(centers)]
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
    plt.show()

def silhouette_analyze(dataframe, cluster_type='KMeans', n_clusters=None):
    """
    Plot silhouette analysis plot of given data and cluster type across different  cluster sizes
    """
    # Use clustering algorithms from here
    # http://scikit-learn.org/stable/modules/clustering.html#clustering
    # And add a plot that visually shows the effectiveness of the clusters/clustering rule.(may be
    # coloured area plots ??)
    from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift, \
                        Birch, AffinityPropagation, AgglomerativeClustering
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import collections
    if not n_clusters:
        n_clusters = range(2, 8, 2)
    assert isinstance(n_clusters, collections.Iterable), "n_clusters must be an iterable object"
    dataframe = dataframe.as_matrix()
    cluster_scores_df = pd.DataFrame(columns=['cluster_size', 'silhouette_score'])
    # Silhouette analysis --
    #       http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    #TODO: Add more clustering methods/types like say dbscan and others

    for j, cluster in enumerate(n_clusters):
        if cluster_type == 'KMeans':
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=cluster, random_state=10)
        elif cluster_type == 'spectral':
            clusterer = SpectralClustering(n_clusters=cluster,
                                                  eigen_solver='arpack',
                                                  affinity="nearest_neighbors")
        elif cluster_type == 'dbscan':
            clusterer = DBSCAN(eps=0.2)

        elif cluster_type == 'birch':
            clusterer = Birch(n_clusters=cluster)

        elif cluster_type == 'affinityProp':
            clusterer = AffinityPropagation(damping=0.9, preference=-200)

        elif cluster_type == 'agglomerativeCluster':
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(dataframe, n_neighbors=10, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            clusterer = AgglomerativeClustering(n_clusters=cluster, linkage='ward',
                                                connectivity=connectivity)
        elif cluster_type == 'meanShift':
            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(dataframe, quantile=0.3)
            clusterer = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
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
        if cluster_labels > 1:
            silhouette_avg = silhouette_score(dataframe, cluster_labels)
            cluster_scores_df.loc[j] = [cluster, silhouette_avg]
            print("For clusters =", cluster,
                    "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(dataframe, cluster_labels)

            y_lower = 10
            for i in range(cluster):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.spectral(float(i) / len(n_clusters))
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
        colors = cm.spectral(cluster_labels.astype(float) / cluster)
        ax2.scatter(dataframe[:, 0], dataframe[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        if hasattr(clusterer, 'cluster_centers_'):
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
                        "with clusters = %d" % (cluster_type, cluster)),
                        fontsize=14, fontweight='bold')

        plt.show()

    plotter.lineplot(cluster_scores_df, xcol='cluster_size', ycol='silhouette_score')

def som_analyze(dataframe, mapsize, algo_type='som'):
    import sompy
    som_factory = sompy.SOMFactory()
    data = dataframe.as_matrix()
    assert isinstance(mapsize, tuple), "Mapsize must be a tuple"
    sm = som_factory.build(data, mapsize= mapsize, normalization='var', initialization='pca')
    if algo_type == 'som':
        sm.train(n_job=6, shared_memory='no', verbose='INFO')

        # View map
        v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)
        v.show(sm, what='codebook', cmap='jet', col_sz=6) #which_dim=[0,1]
        v.show(sm, what='cluster', cmap='jet', col_sz=6) #which_dim=[0,1] defaults to 'all',

        # Hitmap
        h = sompy.hitmap.HitMapView(10, 10, 'hitmap', text_size=8, show_text=True)
        h.show(sm)

    elif algo_type == 'umatrix':
        #But Umatrix finds the clusters easily
        u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
        #This is the Umat value
        UMAT  = u.build_u_matrix(som, distance=1, row_normalized=False)
        u.show(som, distance2=1, row_normalized=False, show_data=True, contooor=False, blob=False)
    else:
        raise "Unknown SOM algorithm type"

def chaid_tree(dataframe, targetCol):
    import CHAID as ch
    columns = dataframe.columns
    columns = list(filter(lambda x: x not in [targetCol], dataframe.columns))
    print(ch.Tree.from_pandas_df(dataframe, columns, targetCol))
