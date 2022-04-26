#!/bin/bash
for i in {1..4}; do
    dir = 'logs/$(date "+%Y/%m/%d %H:%M:%S")'
    echo $dir
    echo '$i/4'
    python gan.py $dir $i
done