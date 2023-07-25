#!/bin/bash
module load tools Apptainer
mkdir /dev/shm/intexml4 -p
TMP_BINDPATH=$SINGULARITY_BINDPATH
SINGULARITY_BINDPATH=
apptainer build $1 $2
$SINGULARITY_BINDPATH=$TMP_BINDPATH
