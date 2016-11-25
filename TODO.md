## TODO
	* Refactor out matplotlib dependencies present in clusteringModels and predictiveModels
	  modules, replacing with calls to the plotter module(which wraps bokeh and seaborn)
	* Add a dask/airflow/luigi/pinball support for training models with different samples on
	  distributed systems.
	* Cleanup the dist_analyze output clubbing the violin plots
	* Cleanup/refactor the plotter.py to remove obsolete/unused plots
	* Add support for feature filtering..(tsfresh module and also others) in features.py
	* Add Gini Coefficient-like measure visual for the cluster analyze
	* Add some test cases
	* Add support for https://github.com/ANNetGPGPU/ANNetGPGPU in the cluster analyze logic
