#!/bin/bash

source /home-nfs/ttran/transitory/speech-nlp/venv_projects/tf_r12/bin/activate
source /home-nfs/ttran/environ

cd /share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/cnn_one2one_pause

DATA_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_conv
MODEL_DIR=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/models

LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_model.py -run_id 4000 -use_speech -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -num_filters 2 -filter_sizes 5-50 -max_epochs 10 -conv_filter 10 -conv_channel 2 -esize 64 -psize 4 -text_hsize 64 -parse_hsize 64 -num_check 10  
#>> $MODEL_DIR/debugf4_maxpool.txt

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_model_pitch3.py -run_id 3000 -use_speech -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -num_filters 2 -filter_sizes 5-50 -max_epochs 10 -conv_filter 10 -conv_channel 2 -esize 64 -psize 4 -text_hsize 64 -parse_hsize 64 -num_check 10 
#>> $MODEL_DIR/debugp3_maxpool.txt

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_model.py -run_id 4001 -multipool -use_speech -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -num_filters 2 -filter_sizes 5-50 -max_epochs 10 -conv_filter 10 -conv_channel 2 -esize 64 -psize 4 -text_hsize 64 -parse_hsize 64 -num_check 10 
#>> $MODEL_DIR/debugf4_multipool.txt

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_model_pitch3.py -run_id 3001 -multipool -use_speech -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -num_filters 2 -filter_sizes 5-50 -max_epochs 10 -conv_filter 10 -conv_channel 2 -esize 64 -psize 4 -text_hsize 64 -parse_hsize 64 -num_check 10 
#>> $MODEL_DIR/debugp3_multipool.txt




#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_model.py -dump_vars -run_id 92 -use_speech -text_num_layers 3 -parse_num_layers 3 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -num_filters 256 -max_epochs 75 

