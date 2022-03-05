#!/bin/bash
n=$1
for i in {1..$n}; do
    echo $(date +"%T")
    echo "$i/$n"
    python gan.py $i $2 png
done