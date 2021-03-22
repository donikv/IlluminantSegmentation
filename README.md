# YEAR 1: Track 1
# Segmentation

## Project Structure

#### Dependancies:  
* Tensorflow 2.4

All of the other dependancies are listed in the exported env.yml file of the anaconda environment. 
To create this environment use `conda env create -f env.yml`. (requires anaconda or miniconda) 

#### Structure:
* general -> general purpose utility functions and classes used by all other classes
	- training -> losses and metrics definitions
	- processing -> image preprocessing and augmentation
	- utils -> general purpose utility funcitons
	- datasets -> definitions of all dataset functions to load various models
	- model -> definitions and classes for parsing custom models loaded from configuration (see [Model definitions](#model-definitions))
* Segmentation -> Training and testing of segmentation models
	- patch -> patch based segmentation
	- gmm -> guassian mixture model segmentation
	- nn -> code to load and train/test cnn segmentation models from custom configurations
	- segmentation\_estimation\_frameworks code to load and train/test cnn segmentation models from custom configurations

## [Model definitions](#model-definitions)
-----
Deep Models are defined using simple custom configuration files. The config files are used with the `CustomModel` class to create tensorflow models. The list of all implemented custom layers can be found in the `CustomModel.py` file. The definitions of models can be found in the `custom_models` folder. Pretrained and trained models can be found in `pretrained` and `training` folders respectively. 
*****

Training configuration is also done using custom config files. These files are responsible for defining which model is trained, on which dataset using what metrics and what loss function, which optimizer etc.


## Training

### Patch based classification:
To train the random forrest patch classifier run segmentation/patch/main.py script with arguments: `dataset_path` -> path to the dataset, `dataset_list` -> name of the list of images to be used for training/validation/testing (0.64/0.16/0.2). The model will be saved to models/classifier/YYYYMMDD-HHMM_rf.joblib

### Deep neural networks:

Procedure for training new models using custom configurations:
1. Define a new model using custom configuration files or use one of the current model configurations
2. Create a new training configuration or change one of the existing ones. In order to use existing training configurations as is, you must change the dataset_path entries in [train] and [valid] sections of the config.
3. 
	* To train deep segmentation models use `unet_custom_train.py` script with arguments:
		- `config_folder` -> path to the training configuration folder
	* To train one of the frameworks use `segmentation_estimation_frameworks/main.py` script with arguments:
		- `config_folder` -> path to the training configuration folder
		- `config_name` -> name of the configuration file
4. The models will be saved in the `<config_folder>/models/YYYYMMDD-HHMM_<config_name>` folder 

## Testing

To test the random forrest patch classifier run `segmentation/patch/test.py` script with arguments: 
- `dataset_path` -> path to the dataset, 
- `dataset_list` -> name of the list of images to be used for testing, 
- `model_definition` -> path to the saved classifier file.
To recreate the reported results call the script:
```
$ python segmenatation/patch/test.py --dataset_path=<PATH TO DATASET FOLDER> --dataset_list=list_outdoor.txt --model_path=models/classifiers/rand_forrest_14_12_all_outdoor.joblib
```

To test the gmm segmentation run `segmenation/gmm/test_clf_hist_seg.py` with arguments: 
- `dataset_path` -> path to the dataset, 
- `dataset_list` -> name of the list of images to be used for testing. 
This will run the model with Places365 classification of indoor/outdoor scenes and gmm segmentation. To recreate the reported results call the script:

```
$ python segmenatation/patch/test_clf_hist_seg.py --dataset_path=<PATH TO DATASET FOLDER> --dataset_list=list_outdoor.txt
```

Procedure for testing models using custom configurations:
1. Select the configuration used for training and change dataset_path entry in [test] section. Everything else should remain as it was during training.
2. 
	* To test deep segmentation models use `unet_custom_test.py` script with arguments:
		- `config_folder` -> path to the training configuration folder
		- `training_instance` -> name of the saved model folder (usually in format YYYYMMDD-HHMM)
	* To train one of the frameworks use `segmentation_estimation_frameworks/test.py` script with arguments:
		- `config_folder` -> path to the training configuration folder
		- `training_instance` -> name of the configuration file
		- `model_definition` -> name of the saved model folder (usually in format YYYYMMDD-HHMM _`config_name`)
3. The statistics for the dice metrics (and angular distance between illuminants for se frameworks) will be saved in the `config_folder`/results/YYYYMMDD-HHMM_`config_name` folder 

*****
To recreate results that are described in the progress report and presentations, use the provided trained models (there is one trained model for each reported result for each type of model/framework), with changing only the `dataset_path` in `[train]`, `[valid]` and `[test]` sections of the training configuration files.
