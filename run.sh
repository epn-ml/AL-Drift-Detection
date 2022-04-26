#!/bin/bash
for i in {1..4}; do
    log="logs/$(date '+%Y-%m-%d_%H-%M-%S')_set$i"
    echo $log
    echo "$i/4"
    python gan.py $log $i $1
done