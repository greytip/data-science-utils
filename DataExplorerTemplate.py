
# coding: utf-8

# In[ ]:

# Custom libraries
from datascienceutils import plotter
from datascienceutils import analyze

# Standard libraries
import json
get_ipython().magic('matplotlib inline')
import datetime
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


# In[ ]:

irisDf = pd.read_csv('./data/Iris.csv')
# Sample Timeseries  picked from here https://www.backblaze.com/b2/hard-drive-test-data.html
hdd2013Df = pd.read_csv('./data/hdd_2013-11-26.csv')


# In[ ]:

# Create classes for showing off correlation_analyze's heatmapping ability
def createClasses(x):
    rdm = random.random()
    if rdm < 0.3:
        return 'A'
    elif rdm > 0.3 and rdm < 0.6:
        return 'B' 
    else:
        return 'C'
irisDf['Class'] = irisDf['Species'].apply(createClasses)


# In[ ]:

irisDf.describe()


# In[ ]:

irisDf.head()


# In[ ]:

irisDf.corr()


# In[ ]:

irisDf.select_dtypes(include=[np.number]).columns


# In[ ]:

analyze.correlation_analyze(irisDf, exclude_columns='Id', 
                                categories=['Species', 'Class'], 
                                measure=['SepalLengthCm','SepalWidthCm',
                                           'PetalLengthCm', 'PetalWidthCm'])


# In[ ]:

analyze.dist_analyze(irisDf, 'SepalLengthCm')


# In[ ]:

analyze.regression_analyze(irisDf, 'SepalLengthCm', 'SepalWidthCm')


# In[ ]:

target = irisDf.Species
irisDf.drop(['Species', 'Class'], 1, inplace=True)


# In[ ]:

analyze.cluster_analyze(irisDf, cluster_type='dbscan')


# In[ ]:

#analyze.som_analyze(df, (10,10), algo_type='som')


# In[ ]:

hdd2013Df.fillna(value=0, inplace=True)
hdd2013Df.describe()


# In[ ]:

hdd2013Df.head()


# In[ ]:

hdd2013Df['date'] = hdd2013Df['date'].astype('datetime64[ns]')


# In[ ]:

hdd2013Df['date'] = [each + datetime.timedelta(0, i*45) for i, each in enumerate(hdd2013Df.date)]


# In[ ]:

analyze.time_series_analysis(hdd2013Df, timeCol='date', valueCol='smart_1_raw', seasonal={'freq': '30s'})


# In[ ]:



