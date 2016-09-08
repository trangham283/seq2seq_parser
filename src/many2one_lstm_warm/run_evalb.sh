#!/bin/bash

step=18291

evalb_path='/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file='/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

gold_file=/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-many2one-lstm-0906/gold-step${step}.txt
br_file=/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-many2one-lstm-0906/decoded-br-step${step}.txt
mx_file=/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-many2one-lstm-0906/decoded-mx-step${step}.txt

$evalb_path -p $prm_file $gold_file $br_file > out_br.txt
$evalb_path -p $prm_file $gold_file $mx_file > out_mx.txt
