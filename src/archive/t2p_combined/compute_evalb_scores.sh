#!/bin/bash

for step in `seq 94000 250 95000`;
do
    echo $step
    python decode_parse_nn.py $step
done
