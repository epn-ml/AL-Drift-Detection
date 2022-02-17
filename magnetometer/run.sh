#!/bin/bash
for n in {1..$1}; do
    python gan.py 1234 0123 png
done