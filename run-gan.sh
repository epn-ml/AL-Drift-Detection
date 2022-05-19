#!/bin/bash
log="logs/gan_$(date '+%Y-%m-%d_%H-%M-%S')"
echo $log
for i in {1..5}; do
    echo "$i/5"
    python gan.py $log $i $1
done