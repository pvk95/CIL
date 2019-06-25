#!/bin/bash

#Train the model first
python3 scripts/main.py -save_folder UNet/ -arch unet -lr 0.01 -batch_sz 1 -epochs 150 -rec_mode 1

#Get predictions
python3 scripts/results.py -save_folder UNet/ -arch unet 

