# Welcome

## Running the code

### FCN8 
- Make sure the training and test data are within the folder called FCN8.
- Test data should be in fcn8/data/test_images
- Training data should be in fcn8/data/training/
- run train.py to train the model and predict on the test images

### Baseline CNN
- Make sure the training and test data are within the folder called baseline_cnn.
- Test data should be in baseline_cnn/data/test_images
- Training data should be in fcn8/data/training/
- run train.py to train the model and predict on the test images

### Training & Prediction

To run training and prediction one has to run the following command:  `./scripts/run_model.sh`

This executes two lines of code 
```bash
#Train the model first
python3 scripts/main.py -save_folder <save_folder>/ -arch <architecture> -lr <learning_rate> -batch_sz <batch_size> -epochs <epochs> -rec_mode <output_creation modes>

#Get predictions
python3 scripts/results.py -save_folder <save_folder>/ -arch <architecture>
```
Where main.py execute the training and the prediction, results.py creates some additional output. The inputs look like this

 * `<save_folder>`: folder where results get saved
 * `<architecture>` : Choose one of architectures you like to train the following, segnet, unet, combined, fcn8, restnnet
 * `<learning_rate>:` Set the learning rate for training. LR of 0.001 is suggested
 * `<batch_size>`: batch size for training. As high as the GPU can handle
 * `<epochs>`: number of epochs to train the network. Depending on the network something between [100, 150, 200]
 * `<ouput_creation_mode>`: chose one between 0-2, on how the stitching together of images should work. 1 seems to perform the best

The `<architecture>` and the `<save_folder>` should be the same in both commands. 

#### output_creation_mode

One of the problem was the different size of train-images and test-images. The `<ouput_creation_mode>` explains how we dealt with that discrepancy.

* 0: Resizing the testing images: The model processes images of size 400x400 and the test images had do be downscaled to be processed before upscaling them again to make a submission.
 
* 1: Patching together with overlap: The model processes 400x400 sized inputs. Test images were split into 400x400 overlapping patches. We created five patches in each direction so a total of 25. The median was taken to recover predictions for the original image.
 
* 2 Patching together without overlap: The model processes 400x400 sized images. The test images are padded with zeros to the size of 800x800. This allowed us to split that image into four sub-images of size 400x400, on which we made the predictions which we stitched together again.

## Cleaning the ouput
If you want to clean the ouput from noise using the morphological operation called opening, you have to run following script.

```bash
python3 scripts/clean_output.py <save_folder> <output_folder>
```

`<save_folder>` points to the save_folder you specified in step one. Whereas `<output_folder>` should be a new folder. 

## Generate Predictions for Uploading

To generate the submission-file for kaggle. One has to first have a folder with a sub-folder called `pred_imgs`. This folder structure is either generated by running the trainining and predictions operations explained above or by the cleaning operation which is also explained above.

The command that creates the submission file, looks like this

```
python3 scripts/mask_to_submission.py <output_folder>/ <ouput_file>
```
Where `<output_folder>` has the structure as explained above and `<output_file>` is the filename of the created submission-file.

The generated submission-file can be found under `<output_folder>/<ouput_file>`
