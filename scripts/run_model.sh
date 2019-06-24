#!/bin/bash

#Train the model first
python3 scripts/main.py -save_folder Combined/ -arch combined -lr 0.01 -batch_sz 32 -epochs 15 -rec_mode 1

#Get predictions
python3 scripts/results.py -save_folder Combined/ -arch combined

