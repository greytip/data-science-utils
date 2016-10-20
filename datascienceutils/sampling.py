from random import gauss, triangular, choice, vonmisesvariate, uniform
from statistics import mean

import io
import random
import sys

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

def reservoir_sampler(filename, sampleThres):
    sample = []
    with open(filename) as f:
        for n, line in enumerate(f):
            if n < sampleThres:
                sample.append(line.rstrip())
            else:
                r = random.randint(0, n)
                if r < sampleThres:
                    sample[r] = line.rstrip()
    return sample

def constant_width_file_sampler(filename, sampleSize):
    sample = []
    with io.open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        random_set = sorted(random.sample(range(filesize), sampleSize))
        for i in range(sampleSize):
            f.seek(random_set[i])
            # Skip current line (because we might be in the middle of a line)
            f.readline()
            # Append the next line to the sample set
            sample.append(f.readline().rstrip())
    return sample

def file_split(filename, sampleSize=50000, k=20):
    fname, ext = filename.split('.')
    with io.open(filename, 'rb') as stream:
        header = stream.readline()
        for i in range(k):
            res = [str(header) + '\n']
            #stream.seek(0, i*sampleSize)
            j = 0
            while j < sampleSize:
                line = stream.readline()
                # Skip a line since we may be in the middle of a line
                line = stream.readline()
                res.append(str(line) + '\n')
                j += 1
            new_fnam = '_'.join([fname, 'sample', str(i) + '.']) + ext
            with io.open(new_fnam, 'w') as fd:
                fd.writelines(list(map(str, res)))

def reservoir_sample_stream(filename, sampleSize):
    res = []
    with io.open(filename, 'r') as stream:
    	for i, el in enumerate(stream):
    	    if i <= sampleSize:
    	        res.append(el)
    	    else:
    	        rand = random.sample(range(i), 1)[0]
    	        if rand < sampleSize:
    	            res[random.sample(range(sampleSize), 1)[0]] = el
    return res


if __name__ == '__main__':
    file_split('/home/anand/DataScientist/kaggle/bosch/data/train_numeric.csv', sampleSize=100, k=2)
