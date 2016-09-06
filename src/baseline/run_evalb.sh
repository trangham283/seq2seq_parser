#!/bin/bash

evalb_path='/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file='/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

$evalb_path -p $prm_file gold_dev.txt baseline_dev.txt
