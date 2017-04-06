#!/bin/bash

source /home-nfs/ttran/transitory/speech-nlp/venv_projects/tf_r12/bin/activate
source /home-nfs/ttran/environ

cd /share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/cnn_one2one_opt

DATA_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_conv
MODEL_DIR=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/models
BEST_DIR=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/best_models


##########################################
# CHANGE train_model.py to eval_model.py 
##########################################

## text-only
#echo id1000 >> results.test.txt 
#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1000 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt 
#
#echo id1100 >> results.test.txt
#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1100 -use_pause -psize 4 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt 
#
#echo id1102 >> results.test.txt
#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1102 -use_pause -psize 16 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt 
#
#echo id1104 >> results.test.txt
#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1104 -use_pause -psize 32 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt 
#
#echo id1200 >> results.test.txt
#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1200 -use_wd -use_pause -psize 4 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt 
#
#echo id1202 >> results.test.txt
#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1202 -use_wd -use_pause -psize 16 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt
#
#echo id1300 >> results.test.txt
#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1300 -use_speech -num_filters 64 -filter_sizes 10-25-50 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt
#
#echo id1350 >> results.test.txt
#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1350 -use_speech -multipool -num_filters 64 -filter_sizes 10-25-50 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt 

echo
echo id1204 >> results.test.txt 
LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1204 -use_wd -use_pause -psize 32 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt

echo 
echo id1400 >> results.test.txt
LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1400 -use_speech -num_filters 64 -filter_sizes 5-15-25-40-50 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt

echo 
echo id1500
LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1500 -use_pause -psize 16 -use_wd -use_speech -num_filters 64 -filter_sizes 10-25-50 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt

echo 
echo id1510
LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1510 -use_pause -psize 16 -use_speech -num_filters 64 -filter_sizes 10-25-50 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt

echo 
echo id1502
LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python eval_model.py -run_id 1502 -use_pause -psize 32 -use_wd -use_speech -num_filters 64 -filter_sizes 10-25-50 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 -test >> results.test.txt





