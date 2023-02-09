#!/bin/bash

OUTPUT=/Users/arsenchik/Desktop/dipploma/machine_learning_in_hft/algotrading/results

EXECUTABLE=$1
TIMEDELTA=$2
EXP=$3

python $EXECUTABLE  --output-dir-path $OUTPUT --time-delta-days $TIMEDELTA --strategy-id $EXP
