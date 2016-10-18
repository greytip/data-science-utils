from random import gauss, triangular, choice, vonmisesvariate, uniform
from statistics import mean

import sys
import random

def posint(x): "Positive integer"; return max(0, int(round(x)))

def SC(): return posint(gauss(15.1, 3) + 3 * triangular(1, 4, 13)) # 30.1
def KT(): return posint(gauss(10.2, 3) + 3 * triangular(1, 3.5, 9)) # 22.1
def DG(): return posint(vonmisesvariate(30, 2) * 3.08) # 14.0
def HB(): return posint(gauss(6.7, 1.5) if choice((True, False)) else gauss(16.7, 2.5)) # 11.7
def OT(): return posint(triangular(5, 17, 25) + uniform(0, 30) + gauss(6, 3)) # 37.0

def repeated_hist(rv, bins=10, k=100000):
    "Repeat rv() k times and make a histogram of the results."
    samples = [rv() for _ in range(k)]
    plt.hist(samples, bins=bins)
    return mean(samples)

def reservoir_sampler(filename, k):
    sample = []
    with open(filename) as f:
        for n, line in enumerate(f):
            if n < k:
                sample.append(line.rstrip())
            else:
                r = random.randint(0, n)
                if r < k:
                    sample[r] = line.rstrip()
    return sample

def constant_width_file_sampler(filename, k):
    import random
    sample = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        random_set = sorted(random.sample(range(filesize), k))
        for i in range(k):
            f.seek(random_set[i])
            # Skip current line (because we might be in the middle of a line)
            f.readline()
            # Append the next line to the sample set
            sample.append(f.readline().rstrip())
    return sample


def reservoir_sample_stream(filename, n):
    import io
    res = []
    with io.open(filename, 'r') as stream:
    	for i, el in enumerate(stream):
    	    if i <= n:
    	        res.append(el)
    	    else:
    	        rand = random.sample(range(i), 1)[0]
    	        if rand < n:
    	            res[random.sample(range(n), 1)[0]] = el
    return res
