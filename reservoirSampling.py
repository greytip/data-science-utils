#!/usr/bin/python
import sys
import random

def reservoirSampler(input, n):
    sample = [];
    for i,line in enumerate(input):
        if i < n:
            sample.append(line)
        elif i >= n and random.random() < n/float(i+1):
            replace = random.randint(0,len(sample)-1)
            sample[replace] = line

    for line in sample:
        sys.stdout.write(line)

def test(conn, orig_table_name, reservoir_name=None):
    pass

def main():
    if len(sys.argv) == 3:
        input = open(sys.argv[2],'r')
    elif len(sys.argv) == 2:
        input = sys.stdin;
    else:
        sys.exit("Usage:  python samplen.py <lines> <?file>")
    N = int(sys.argv[1]);
    reservoirSampler(input, N)
