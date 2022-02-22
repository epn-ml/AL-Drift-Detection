#!/bin/bash
for n in {1..4}; do
    echo $(date +"%T")
    echo "$n/4"
    python gan.py $n $1 png
done