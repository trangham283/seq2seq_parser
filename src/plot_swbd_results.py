#!/usr/bin/env python

# Process results from training

import os
import sys
import argparse
import random
import re
import matplotlib.pyplot as plt
import numpy as np

#pa = argparse.ArgumentParser(description='Process result file')
#pa.add_argument('--result_file', help='result filename')
#args = pa.parse_args()

#filename = args.result_file
# debug: 
filename = 'toplot_swbd_0801.txt'
raw_lines = open(filename).readlines()
slines = [x.strip() for x in raw_lines]

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
    if "Bracket only" in line:
        br_s = float(slines[i].split()[6].rstrip(';'))
        br_sents.append(br_s)
        br_f = float(slines[i].split()[-1].rstrip(';'))
        br_f1.append(br_f)
    if "Matched XX" in line:
        mx_s = float(slines[i].split()[6].rstrip(';'))
        mx_sents.append(mx_s)
        mx_f = float(slines[i].split()[-1].rstrip(';'))
        mx_f1.append(mx_f)


# plot perplexities
plt.figure()
#plt.plot(step_num, b1_ppl, 'b*-', step_num, b2_ppl, 'g*-', 
#         step_num, b3_ppl, 'r*-', 
#         step_num, np.array([b1_ppl, b2_ppl, b3_ppl]).mean(axis=0), 'ko-',
#         step_num, train_ppl, 'k:')
#plt.legend(['Devset Bucket (10, 40)','Devset Bucket (25, 85)', 
#'Devset Bucket (40, 150)', 'Devset Average', 'Trainset'])
plt.plot(step_num, np.array([b1_ppl, b2_ppl, b3_ppl]).mean(axis=0), 'k.-',
         step_num, train_ppl, 'b:')
plt.legend(['Devset Average', 'Trainset'])

plt.grid(True)
plt.xlabel('Step')
plt.ylabel('Perplexity')
plt.title('Training losses: swbd_0801')

# plot F1
plt.figure()
plt.subplot(2,1,1)
plt.plot(step_num, br_sents, 'bo--', step_num, mx_sents, 'ro--')
plt.grid(True)
plt.xlabel('Step')
plt.ylabel('# valid sentences')
#plt.legend(['Bracket-only', 'XX-help'], loc=4)
plt.title('EVALB results: swbd_0801')

plt.subplot(2,1,2)
plt.plot(step_num, br_f1, 'bo-', step_num, mx_f1, 'ro-')
plt.grid(True)
plt.xlabel('Step')
plt.ylabel('F1 score')
plt.legend(['Bracket-only', 'XX-help'], loc=4)
#plt.title('EVALB results '+ filename.strip('output_').strip('.txt'))

# plot F1
plt.figure()
plt.subplot(2,1,1)
plt.plot(step_num, br_sents, 'bo--')
plt.grid(True)
plt.ylabel('# valid sentences')
plt.title('EVALB results: bracket_only; swbd_0801')

plt.subplot(2,1,2)
plt.plot(step_num, br_f1, 'bo-')
plt.grid(True)
plt.xlabel('Step')
plt.ylabel('F1 score')
#plt.legend(['Bracket-only', 'XX-help'])
#plt.title('EVALB results '+ filename.strip('output_').strip('.txt'))

