#!/bin/bash
log="logs/cnn_$(date '+%Y-%m-%d_%H-%M-%S')"
echo $log
for i in {1..7}; do
    echo "$i/7"
    python cnn.py $log $i $1
done