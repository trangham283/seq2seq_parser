#!/bin/bash

for step in `seq 10500 500 11500`;
do
    echo $step
    python decode_parse_nn.py $step
done
