#!/bin/bash

#BASE_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models
BASE_DIR=/s0/ttmt001


#MODEL_DIR=lr_0.001_text_hsize_128_text_num_layers_2_speech_hsize_128_speech_num_layers_2_parse_hsize_128_parse_num_layers_2_num_filters_5_filter_sizes_10-25-50_out_prob_0.8_run_id_0
#STEP_NUM=42444

#MODEL_DIR=lr_0.001_text_hsize_128_text_num_layers_2_speech_hsize_128_speech_num_layers_2_parse_hsize_128_parse_num_layers_2_num_filters_5_filter_sizes_10-25-50_out_prob_0.7_run_id_1
#STEP_NUM=28240

#MODEL_DIR=lr_0.001_text_hsize_128_text_num_layers_3_speech_hsize_128_speech_num_layers_3_parse_hsize_128_parse_num_layers_3_num_filters_5_filter_sizes_10-25-50_out_prob_0.8_run_id_3
#STEP_NUM=64952

#MODEL_DIR=lr_0.001_text_hsize_256_text_num_layers_2_speech_hsize_256_speech_num_layers_2_parse_hsize_256_parse_num_layers_2_num_filters_5_filter_sizes_10-25-50_out_prob_0.7_run_id_0
#STEP_NUM=25721

#MODEL_DIR=lr_0.001_text_hsize_256_text_num_layers_2_speech_hsize_256_speech_num_layers_2_parse_hsize_256_parse_num_layers_2_num_filters_10_filter_sizes_10-25-50_out_prob_0.7_run_id_0
#STEP_NUM=44364

#MODEL_DIR=lr_0.001_text_hsize_256_text_num_layers_3_speech_hsize_256_speech_num_layers_3_parse_hsize_256_parse_num_layers_3_num_filters_5_filter_sizes_10-25-50_out_prob_0.7_run_id_0
#STEP_NUM=59346
#STEP_NUM=40977

# Embedding size = 512 here
MODEL_DIR=lr_0.001_text_hsize_256_text_num_layers_2_speech_hsize_256_speech_num_layers_2_parse_hsize_256_parse_num_layers_2_num_filters_5_filter_sizes_10-25-50_out_prob_0.7_run_id_2
STEP_NUM=40977

#MODEL_DIR=lr_0.001_text_hsize_256_text_num_layers_2_speech_hsize_256_speech_num_layers_2_parse_hsize_256_parse_num_layers_2_num_filters_15_filter_sizes_10-25-50_out_prob_0.7_run_id_2
#STEP_NUM=40977

#MODEL_DIR=lr_0.001_text_hsize_256_text_num_layers_2_speech_hsize_256_speech_num_layers_2_parse_hsize_256_parse_num_layers_2_num_filters_5_filter_sizes_10-25-50_out_prob_0.8_run_id_2
#STEP_NUM=19768
#STEP_NUM=5648
#STEP_NUM=2824

# model with zero speech, same architecture as model above
#MODEL_DIR=run_id_98
#STEP_NUM=31064

# text-only, with same parameters as above
#MODEL_DIR=text_only
#STEP_NUM=43803

#MODEL_DIR=text_only_prevbest
#STEP_NUM=21917

GOLD=$BASE_DIR/$MODEL_DIR/gold-step${STEP_NUM}.txt
BR=$BASE_DIR/$MODEL_DIR/decoded-br-step${STEP_NUM}.txt
MX=$BASE_DIR/$MODEL_DIR/decoded-mx-step${STEP_NUM}.txt

# print_eval_prf.py <gold> <test> <output_prefix>
#python print_eval_prf.py $GOLD $BR $BASE_DIR/$MODEL_DIR/results-br-step${STEP_NUM}.txt
python print_eval_prf.py $GOLD $MX $BASE_DIR/$MODEL_DIR/results-mx-step${STEP_NUM}.txt

tail -n2 $BASE_DIR/$MODEL_DIR/results-mx-step${STEP_NUM}.txt.out
grep length $BASE_DIR/$MODEL_DIR/results-mx-step${STEP_NUM}.txt.err

