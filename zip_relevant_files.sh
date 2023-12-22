#!/bin/sh


now=$(date +"%d_%m_%Y")
printf $now
zip -r "$PWD/PaderbornUniversity_submission_$now.zip" \
    Model/data/*.json \
    Model/data/output/*meta* \
    Model/data/models/*c9cfe* \
    Model/data/input/test/place_Testing_here.txt \
    Model/src/Model_Inference.py \
    Model/src/run_cnn_*.py \
    Model/src/utils/*py \
    Result/ \
    README.md \
    requirements.txt \
    PaderbornUniversity_Report.pdf

