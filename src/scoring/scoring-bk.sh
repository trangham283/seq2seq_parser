#!/bin/bash

SPLIT="test"
DATA_DIR=/homes/ttmt001/transitory/seq2seq_parser/data
GOLD=$DATA_DIR/${SPLIT}_trees_for_bk_new_buckets.mrg
TEST=$DATA_DIR/${SPLIT}_bk_compat.out

# print_eval_prf.py <gold> <test> <output_prefix>
python print_eval_prf.py $GOLD $TEST $DATA_DIR/${SPLIT}_bk_results


