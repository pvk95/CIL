#!/bin/bash

#Train the model first
python scripts/main.py -save_folder ResUNet/

#Get predictions
python scripts/results.py -save_folder ResUNet/

