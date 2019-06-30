#!/bin/bash

#Train the model first
python3 scripts/main.py -save_folder FCN8/ -arch FCN8 -lr 0.001 -batch_sz 1 -epochs 15 -rec_mode 1

#Get predictions
python3 scripts/results.py -save_folder FCN8/ -arch FCN8

