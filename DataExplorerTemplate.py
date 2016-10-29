
# coding: utf-8

# In[ ]:

# Custom libraries
from datascienceutils import plotter
from datascienceutils import analyze

# Standard libraries
import json
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

# Create classes for showing off correlation_analyze's heatmapping ability
def createClasses(x):
    if x == 'Iris-setosa':
        return 'A'
    elif x == 'Iris-versicolor':
        return 'B'
    else:
        return 'C'
df['Class'] = df['Species'].apply(createClasses)


# In[ ]:

df.head()


# In[ ]:

df.corr()


# In[ ]:

df['Species'].unique()


# In[ ]:

analyze.correlation_analyze(df, exclude_columns='Id', categories=['Species', 'Class'], measure=['SepalLengthCm', 
                                                                                      'SepalWidthCm',
                                                                                       'PetalLengthCm',
                                                                                       'PetalWidthCm'])


# In[ ]:

df.columns
target = df.Species
df.drop('Species', 1, inplace=True)


# In[ ]:

#analyze.time_series_analysis(df, timeCol='date', valueCol='count')


# In[ ]:

analyze.cluster_analyze(df, cluster_type='dbscan')


# In[ ]:

analyze.som_analyze(df, (30,30), algo_type='som')

