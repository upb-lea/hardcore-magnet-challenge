#!/bin/sh


now=$(date +"%d_%m_%Y")
printf $now
zip -r "$PWD/PaderbornUniversity_submission_$now.zip" \
    Model/data/*.json \
    Model/data/input/test/Testing/ \
    Model/data/input/test/Training/ \
    Model/data/output/*meta* \
    Model/data/models/*c9cfe* \
    Model/src/Model_Inference.py \
    Model/src/run_cnn_*.py \
    Model/src/utils/*py \
    Result/ \
    README.md

