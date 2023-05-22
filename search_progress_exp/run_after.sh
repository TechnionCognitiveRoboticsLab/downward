#!/bin/sh

python3 generate_dataset.py > dataset.csv
mkdir odts
python3 generate_odts_exp.py