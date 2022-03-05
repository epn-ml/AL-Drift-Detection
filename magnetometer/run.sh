#!/bin/bash
n=3
for i in {1..$n}; do
    echo $(date +"%T")
    echo "$i/$n"
    python gan.py $2 png
done