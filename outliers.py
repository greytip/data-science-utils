# -*- coding: utf-8
################################################################
#   UniVariate Outliers
##################################################################
import numpy as np
def getOutliers(data, m = 2.):
    """
    data -- is a pandas data frame
    m --- is the number of sigma deviations
    return:
        Simple data points more than given value times the median
    """
    # by median
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s>m]

def sigmaDeviation(seq, threshold=3, passes=1):
    # filter and remove values beyond +/- 3 sigma variance
    # default one pass.
    for i in range(passes):
        std     = np.std(seq)
        seq = filter(lambda x: x > std, seq)
    return seq

def interQuartileRangeDev(seq, threshold=1.5):
    # filter values beyound the +/- 1.5 quartile range
    pass

def capPercentile(seq, threshold=5):
    # filter values from the 5th and 95th percentile range
    pass

def zScoreSpikes(seq, zthresh=2 ):
    mean    = np.mean(data)
    std     = np.std(data)

    o3      = mean + (2) *std
    o4      = mean + (-2)*std
    pass

################################################################
#   Multi/Bi-Variate Outliers
##################################################################
#   -- Measured using an index of influence or leverage or distance
#   -- Popular indices such as Mahalanobis’ distance and Cook’s D are frequently used .
#   --  statistical measure like STUDENT, COOKD, RSTUDENT and others.
###
# Removal methods
#    -- Deleting
#    -- Transforming(like log or others) and binning
#    -- Imputing values

