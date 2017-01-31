#!/bin/bash

# Based on run.sh and local/eval2000_data_prep.sh
maindir=/tmp/ttran/Datasets/audio_swbd
sdir=$maindir/sph

# make list of files to process
find $sdir -iname '*.sph' | sort > sph.flist
sed -e 's?.*/??' -e 's?.sph??' sph.flist | paste - sph.flist > sph.scp

sph2pipe=$HOME/sw/kaldi/tools/sph2pipe_v2.5/sph2pipe
[ ! -x $sph2pipe ] \
  && echo "Could not execute the sph2pipe program at $sph2pipe" && exit 1;

awk -v sph2pipe=$sph2pipe '{
  printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2); 
  printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);
}' < sph.scp | sort > wav.scp || exit 1;
#side A - channel 1, side B - channel 2

awk '{print $1}' wav.scp \
  | perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_";
               print "$1-$2 $1 $2\n"; ' \
  > reco2file_and_channel || exit 1;

fbankdir=$maindir/fbank
nj=8
cmd=utils/run.pl
mfcc_config=conf/mfcc.conf
pitch_config=conf/pitch.conf
fbank_config=conf/fbank.conf
compress=true

data=$sdir
logdir=$maindir/logfbank

# use "name" as part of name of the archive.
name=`basename $data`

swdir=/home-nfs/ttran/sw/kaldi/src/featbin

mkdir -p $fbankdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $maindir/feats.scp ]; then
    mkdir -p $maindir/.backup
    echo "$0: moving $maindir/feats.scp to $maindir/.backup"
    mv $maindir/feats.scp $maindir/.backup
fi

cp wav.scp $maindir/
scp=$maindir/wav.scp
required="$scp $fbank_config"

for f in $required; do
    if [ ! -f $f ]; then
        echo "make_fbank.sh: no such file $f"
        exit 1;
    fi
done

split_scps=""
for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
done

utils/split_scp.pl $scp $split_scps || exit 1;


# You have to use the ",t" modifier on the output, for instance
# copy-feats scp:feats.scp ark,t:-

## compute fbank (+energy) here
$cmd JOB=1:$nj $logdir/make_fbank_${name}.JOB.log \
$swdir/compute-fbank-feats --verbose=2 --config=$fbank_config \
scp,p:$logdir/wav_${name}.JOB.scp ark:- \| \
$swdir/copy-feats --compress=$compress ark:- \
ark,scp:$fbankdir/raw_fbank_$name.JOB.ark,$fbankdir/raw_fbank_$name.JOB.scp \
|| exit 1;

$cmd JOB=1:$nj $logdir/copy_text_${name}.JOB.log \
$swdir/copy-feats ark:$fbankdir/raw_fbank_$name.JOB.ark ark,t:$fbankdir/raw_fbank_$name.JOB.txt \
|| exit 1;

if [ -f $logdir/.error.$name ]; then
  echo "Error producing fbank features for $name:"
  tail $logdir/make_fbank_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $fbankdir/raw_fbank_$name.$n.scp || exit 1;
done > $maindir/feats.scp


rm $logdir/wav_${name}.*.scp  2>/dev/null

echo "Succeeded creating features for $name"

