#!/usr/bin/bash

FILES=`ls *.sph`

for f in ${FILES}
do 
    NAME=`basename -s \.sph ${f}`
    echo ${NAME}
    sox ${f} ${NAME}.wav 
done
