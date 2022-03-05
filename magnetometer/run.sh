#!/bin/bash
for n in {1..$1}; do
    echo $(date +"%T")
    echo "$n/$1"
    python gan.py $n $2 png
done