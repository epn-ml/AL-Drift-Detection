#!/bin/bash
for n in {1..3}; do
    echo "$n/3"
    python gan.py $n $1 png
done