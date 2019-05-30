The ImageClassifier app runs two different CNN models for image recongnition tasks. 

It supports the following models, imported from torchvision:

1. vgg16
2. densenet121

Esentially, the app allows the user to select any of the 2 CNN models. 
It attaches a classifier to the model that the user selects and trains it to correctly classify 102 different types of flowers.
Flower images are found in the "flowers/" directory. 

Images are separated into 3 datasetes : train, validate and test. 

train.py trains the model, on the following parameters, that may be passed by the user through the CLI:

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg16" or --arch "densenet121"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

predict.py allows the user to predict what flower type a particular flower image corresponds to, again, passing the 
following arguments through the CLI:

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu


ALC
