import os
import cPickle as pickle

split = 'dev'
file_name = split + '_sentID.pickle'
d = pickle.load(open(file_name))
for bucket in d:
    for sample in bucket:
        print sample
