#!/bin/bash

#step=14080
step=26356
#step=53066
#step=69962
out_prob=0.8
speech_num_layers=1
id=5
opt='_' # for adam
#opt=_opt_adagrad_

evalb_path='/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file='/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

model_dir=/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/many2one_tune/models
id_dir=lr_0.001_bsize_64_esize_512_text_hsize_500_text_num_layers_2_speech_hsize_500_speech_num_layers_${speech_num_layers}_parse_hsize_500_parse_num_layers_2_out_prob_${out_prob}_run_id_${id}${opt}rnn_lstm_many2one_speech


gold_file=${model_dir}/${id_dir}/gold-step${step}.txt
br_file=${model_dir}/${id_dir}/decoded-br-step${step}.txt
mx_file=${model_dir}/${id_dir}/decoded-mx-step${step}.txt

$evalb_path -p $prm_file $gold_file $br_file > out_br_id_${id}.txt
$evalb_path -p $prm_file $gold_file $mx_file > out_mx_id_${id}.txt
