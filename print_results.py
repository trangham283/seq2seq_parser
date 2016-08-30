#!/usr/bin/env python

# Process results from training

import os
import sys
import argparse
import random
import re
#import matplotlib.pyplot as plt
import numpy as np

pa = argparse.ArgumentParser(description='Process result file')
pa.add_argument('--result_file', help='result filename')
args = pa.parse_args()

filename = args.result_file
# debug: 
#filename = 'output_baseline_0718.txt'

raw_lines = open(filename).readlines()
slines = [x.strip() for x in raw_lines]

# example structure:
# line i: global step 100 learning rate 0.1000 step-time 0.64 perplexity 12.91
# line i+1: eval: bucket 0 perplexity 13.43
# line i+2: eval: bucket 1 perplexity 9.60
# line i+3: eval: bucket 2 perplexity 9.92
# line i+4: Doing evalb
# line i+5: Eval with completed brackets only
# line i+6: ['Number of Valid sentence  =      0', 'Number of Valid sentence  =      0']
# line i+7: ['Bracketing FMeasure       =   -nan', 'Bracketing FMeasure       =   -nan']
# line i+8: Eval with matched XXs and brackets
# line i+9: ['Number of Valid sentence  =   1565', 'Number of Valid sentence  =   1565']
# line i+10: ['Bracketing FMeasure       =  17.49', 'Bracketing FMeasure       =  17.49']

step_num = []
train_ppl = []
b1_ppl = []
b2_ppl = []
b3_ppl = []
br_sents = []
br_f1 = []
mx_sents = []
mx_f1 = []

for i, line in enumerate(slines):
    if "global step" in line:
        this_step = int(line.split()[2])
        ppl = float(line.split()[-1])
        step_num.append(this_step)
        train_ppl.append(ppl)
    if "bucket 0" in line:
        b1 = float(slines[i].split()[-1])
        b1_ppl.append(b1)
    if "bucket 1" in line:
        b2 = float(slines[i].split()[-1])
        b2_ppl.append(b2)
    if "bucket 2" in line:
        b3 = float(slines[i].split()[-1])
        b3_ppl.append(b3)
    if "brackets only" in line:
        br_s = float(slines[i+1].split()[-1].strip('\']'))
        br_sents.append(br_s)
        br_f = float(slines[i+2].split()[-1].strip('\']'))
        br_f1.append(br_f)
    if "XXs and brackets" in line:
        mx_s = float(slines[i+1].split()[-1].strip('\']'))
        mx_sents.append(mx_s)
        mx_f = float(slines[i+2].split()[-1].strip('\']'))
        mx_f1.append(mx_f)

print "Losses"
for i in range(len(step_num)):
    print step_num[i], '\t', train_ppl[i], '\t', b1_ppl[i], '\t', \
            b2_ppl[i], '\t', b3_ppl[i], '\t', np.mean([b1_ppl[i], b2_ppl[i], \
            b3_ppl[i]])


print "EVALB results"
for i in range(len(step_num)):
    print step_num[i], '\t', br_sents[i], '\t', br_f1[i], '\t', \
            mx_sents[i], '\t', mx_f1[i]


