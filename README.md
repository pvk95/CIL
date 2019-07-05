# Welcome

## Running the code

### Training & Prediction

To run training and prediction one has to run the following command:  `./scripts/run_model.sh`

This executes two lines of code 
```bash
#Train the model first
python3 scripts/main.py -save_folder Baseline/ -arch baseline -lr 0.001 -batch_sz 1 -epochs 150 -rec_mode 1

#Get predictions
python3 scripts/results.py -save_folder Baseline/ -arch baseline 
```