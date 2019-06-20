#!/bin/bash

#Train the model first
python3 scripts/main.py -save_folder Combined/ -arch combined -lr 0.01 -batch_sz 1 -epochs 10

#Get predictions
python3 scripts/results.py -save_folder Combined/ -arch combined 

