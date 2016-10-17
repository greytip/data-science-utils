from collections import defaultdict
from itertools import tee
from sqlalchemy import create_engine, MetaData, Table

import numpy as np
import scipy

class ProbDist(dict):
    """
			Taken from norvig: http://nbviewer.jupyter.org/url/norvig.com/ipython/Probability.ipynb
			input: A dict of {<event_id_or_name>: <frequency>}
			return: A Probability Distribution; an {outcome: probability} mapping.
	"""
    def __init__(self, mapping=(), **kwargs):
        self.update(mapping, **kwargs)
        # Make probabilities sum to 1.0; assert no negative probabilities
        total = sum(self.values())
        for outcome in self:
            self[outcome] = self[outcome] / total
            assert self[outcome] >= 0

def init_board_gauss(N, k):
    """
    Taken from https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
    """
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

def measure_skew(val_array):
    return scipy.stats.skewtest(val_array)

def measure_kurtosis(val_array, **kwargs):
    return scipy.stats.kurtosis(val_array, **kwargs)

#Replacement function for scipy.special.erf
def erf(x):
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

def pairwise(seq):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(seq)
    next(b, None)
    return zip(a, b)


def values_dist(vals, binsize=10):
    """
    Return a simple frequency distribution of a given list
    """
    if not vals:
        vals = [0]

    x = np.sort(np.asarray(vals))

    mu, sigma = np.nanmean(x), np.nanstd(x)
    hist,xedges  = np.histogram(x, density=False)
    pdf, cdf = None, None
    #pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    #cdf = (1 + erf((x-mu)/np.sqrt(2*sigma**2)))/2
    return hist, xedges, pdf, cdf

