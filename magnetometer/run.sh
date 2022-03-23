#!/bin/bash
for i in {1..4}; do
    echo $(date +"%T")
    echo "$i/4"
    python gan.py $i $1 png $2
done