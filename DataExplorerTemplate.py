
# coding: utf-8

# In[ ]:

# Custom libraries
from datascienceutils import plotter
from datascienceutils import analyze


# Standard libraries
get_ipython().magic('load_ext autoreload')
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import json
#fig=plt.figure()
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics


from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource
from bokeh.charts import Histogram
import bokeh
output_notebook(bokeh.resources.INLINE)

from sqlalchemy import create_engine


# In[ ]:

df = pd.read_csv('./data/Iris.csv')


# In[ ]:

df.describe()


# In[ ]:

df.head()


# In[ ]:

df.corr()


# In[ ]:

analyze.correlation_analyze(df, exclude_columns='Id')


# In[ ]:

df.columns
target = df.Species
df.drop('Species', 1, inplace=True)


# In[ ]:

#analyze.time_series_analysis(df, timeCol='date', valueCol='count')


# In[ ]:

analyze.silhouette_analyze(df, cluster_type='KMeans', n_clusters=range(2,4))


# In[ ]:

analyze.silhouette_analyze(df, cluster_type='spectral', n_clusters=range(2,5))


# In[ ]:

analyze.cluster_analyze(df, cluster_type='KMeans', n_clusters=4)


# In[ ]:

analyze.cluster_analyze(df, cluster_type='dbscan')


# In[ ]:

analyze.som_analyze(df, (30,30), algo_type='som')


# In[ ]:

df.columns


# In[ ]:

#new_df =df.copy(deep=True)
#new_df.drop('Id', 1, inplace=True)
analyze.chaid_tree(df, 'y_pred')

