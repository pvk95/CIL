#!/bin/bash

#Train the model first
python3 scripts/main.py -save_folder Baseline/ -arch baseline -lr 0.001 -batch_sz 1 -epochs 150 -rec_mode 1

#Get predictions
python3 scripts/results.py -save_folder Baseline/ -arch baseline 


