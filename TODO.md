## TODO
	* Refactor out matplotlib dependencies present in clusteringModels and predictiveModels
	  modules, replacing with calls to the plotter module(which wraps bokeh and seaborn)

	* Add a dask/airflow/luigi/pinball support for training models with different samples on
	  distributed systems.

	* Cleanup/refactor the plotter.py to remove obsolete/unused plots

	* Add support for feature filtering..(tsfresh module and also others) in features.py

	* Add Gini Coefficient-like measure visual for the cluster analyze

	* Add support for https://github.com/ANNetGPGPU/ANNetGPGPU in the cluster analyze logic

	* Add 3D heatmaps and may be 3D + 1D(time) visualizations/Animations(like the gapminder
	  bubble chart for ex:)

    	* analyze TODO: May be add a way to plot joint distributions of two variables?

    	* analyze TODO: add grouped violinplots by categorical variables too.

	* Add a separate grid search function to grid search a data set with the given
	  model.(wrapper around sklearn model_selection's grid search)

	* Add Gaussian Mixture Model to clustering models

	* Add two templates, one analyze, one clusters, and one the regular predictions models

