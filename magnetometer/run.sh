#!/bin/bash
for n in {1..3}; do
    echo "$n/3"
    python gan.py 1234 13 png
done