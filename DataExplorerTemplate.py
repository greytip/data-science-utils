
# coding: utf-8

# In[1]:

# Custom libraries
from datascienceutils import plotter
from datascienceutils import analyze

# Standard libraries
import json
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import random
from sklearn import cross_validation
from sklearn import metrics


from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource
from bokeh.charts import Histogram
import bokeh
output_notebook(bokeh.resources.INLINE)

from sqlalchemy import create_engine


# In[2]:

df = pd.read_csv('./data/Iris.csv')


# In[3]:

# Create classes for showing off correlation_analyze's heatmapping ability
def createClasses(x):
    rdm = random.random()
    if rdm < 0.3:
        return 'A'
    elif rdm > 0.3 and rdm < 0.6:
        return 'B' 
    else:
        return 'C'
df['Class'] = df['Species'].apply(createClasses)


# In[4]:

df.describe()


# In[5]:

df.head()


# In[6]:

df.corr()


# In[7]:

df['Species'].unique()


# In[8]:

analyze.correlation_analyze(df, exclude_columns='Id', 
                                categories=['Species', 'Class'], 
                                measure=['SepalLengthCm','SepalWidthCm',
                                           'PetalLengthCm', 'PetalWidthCm'])


# In[9]:

target = df.Species
df.drop(['Species', 'Class'], 1, inplace=True)


# In[10]:

#analyze.time_series_analysis(df, timeCol='date', valueCol='count')


# In[11]:

analyze.cluster_analyze(df, cluster_type='dbscan')


# In[12]:

#analyze.som_analyze(df, (10,10), algo_type='som')

