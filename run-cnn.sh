#!/bin/bash
log="logs/cnn_$(date '+%Y-%m-%d_%H-%M-%S')"
echo $log
for i in {1..5}; do
    echo "$i/5"
    python cnn.py $log $i $1 $2
done