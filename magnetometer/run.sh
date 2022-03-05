#!/bin/bash
for n in {1..$1}; do
    echo $(date +"%T")
    echo "$n/4"
    python gan.py $n $2 png
done