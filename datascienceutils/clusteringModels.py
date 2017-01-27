from bokeh.layouts import gridplot
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
import numpy as np
import pandas as pd
import time

# Custom utils
from . import sklearnUtils as sku
from . import plotter
from . import utils

#TODO: add a way of weakening the discovered cluster structure and running again
# http://scilogs.spektrum.de/hlf/sometimes-noise-signals/
def is_cluster(dataframe, model_type='dbscan', batch_size=2):
    if model_type == 'dbscan':
        model_obj = cluster.DBSCAN(eps=.2)
    elif model_type == 'MiniBatchKMeans':
        assert batch_size, "Batch size mandatory"
        model_obj = cluster.MiniBatchKMeans(n_clusters=batch_size)
    else:
        pass
    model_obj.fit(X)
    return model_obj.cluster_centers_

def cluster_analyze(dataframe):

    clustering_names = [
    'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
    'SpectralClustering', 'Ward', 'AgglomerativeClustering',
    'DBSCAN', 'Birch']
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)


    plot_num = 1

    # normalize dataset for easier parameter selection
    X = sku.feature_scale_or_normalize(dataframe, dataframe.columns)
    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # create clustering estimators
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
                                           connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=2,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=.2)
    affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                       preference=-200)

    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=2,
        connectivity=connectivity)


    birch = cluster.Birch(n_clusters=2)
    clustering_algorithms = [
        two_means, affinity_propagation, ms, spectral, ward, average_linkage,
        dbscan, birch]
    plots = list()
    for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        # plot
        new_df = pd.DataFrame(X)
        plots.append(plotter.scatterplot(new_df, 0, 1, title='%s'%name))

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            centers_df = pd.DataFrame(centers)
            center_colors = colors[:len(centers)]
            plotter.scatterplot(centers_df, 0, 1,fill_color="r")
    grid = gridplot(list(utils.chunks(plots,size=2)))
    plotter.show(grid)
