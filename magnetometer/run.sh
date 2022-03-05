#!/bin/bash
for i in {1..3}; do
    echo $(date +"%T")
    echo "$i/3"
    python gan.py $1 png
done