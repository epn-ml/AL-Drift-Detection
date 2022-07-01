#!/bin/bash
log="logs/cnn_$(date '+%Y-%m-%d_%H-%M-%S')"
echo $log
for i in {5..25}; do
    echo "max_orbits $i/25"
    python cnn.py $log 7 $1 $i
done