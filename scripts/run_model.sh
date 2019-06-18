#!/bin/bash

#Train the model first
python scripts/main.py -save_folder UNet/ -arch unet -lr 0.1 -batch_sz 64

#Get predictions
python scripts/results.py -save_folder UNet/ -arch unet

