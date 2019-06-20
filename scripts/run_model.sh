#!/bin/bash

#Train the model first
python3 scripts/main.py -save_folder SegNet/ -arch segnet -lr 0.01 -batch_sz 64 -epochs 150

#Get predictions
python3 scripts/results.py -save_folder SegNet/ -arch segnet

