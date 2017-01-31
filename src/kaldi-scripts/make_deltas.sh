#!/bin/bash

feattype='fbank'

# Based on run.sh and local/eval2000_data_prep.sh
maindir=/tmp/ttran/Datasets/audio_swbd
sdir=$maindir/sph

#maindir=/tmp/ttran/Datasets/audio_debug
#sdir=/scratch/ttran/Datasets/audio_debug

rawdir=$maindir/${feattype}
deltadir=$maindir/${feattype}_delta
nj=8
cmd=utils/run.pl
compress=true

data=$sdir
logdir=$maindir/log${feattype}

# use "name" as part of name of the archive.
name=`basename $data`

swdir=/home-nfs/ttran/sw/kaldi/src/featbin

mkdir -p $deltadir || exit 1;

$cmd JOB=1:$nj $logdir/delta_${feattype}_${name}.JOB.log \
$swdir/add-deltas --delta-order=2 --delta-window=2 \
scp,p:$rawdir/raw_${feattype}_${name}.JOB.scp ark:- \| \
$swdir/copy-feats --compress=$compress ark:- \
ark,scp:$deltadir/${feattype}_delta_$name.JOB.ark,$deltadir/${feattype}_delta_$name.JOB.scp \
|| exit 1;

$cmd JOB=1:$nj $logdir/copy_${feattype}_delta_text_${name}.JOB.log \
$swdir/copy-feats ark:$deltadir/${feattype}_delta_$name.JOB.ark ark,t:$deltadir/${feattype}_delta_$name.JOB.txt \
|| exit 1;

#rm $logdir/wav_${name}.*.scp  2>/dev/null

echo "Succeeded creating delta features for $name"

