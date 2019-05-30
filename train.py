#train.py trains a new network on a dataset and saves the model as a checkpoint

#import external dependencies
import torch
from torchvision import models
#importing methods from app
from get_train_args import get_train_args
from process_data import process_data
from classifier import load_model
from train_loop import train_loop
from test import test
#from validation_loop import validation_loop

#main function definition
def main():
    #[[[[    TRAINING PART    ]]]
    #get relevant args for training
    train_args = get_train_args()
    #call process_data.py to get processed data. Pass --data_dir, return loaders.
    trainloader, validationloader, testloader, train_datasets = process_data(train_args.data_dir)
    #call classifier.py to set up model. Pass --arch, --learning_rate, --epochs and --gpu. Return loaded model.
    model = load_model(train_args.arch, train_args.hidden_units)
    print(model)
    #call train_loop.py to train model. Pass loaded model and train and validation loaders. Return trained model.
    trained_model = train_loop(model, trainloader, validationloader,  train_args.learning_rate, train_args.epochs, train_args.gpu)
    print(trained_model)
    
    #test the model
    test(trained_model, testloader, train_args.gpu)
    #save the checkpoint and return it.
    if train_args.arch == "vgg16":
        model_arch = models.vgg16(pretrained=True)
    else:
        model_arch = models.densenet121(pretrained=True)
    
    checkpoint = {
              "model": model_arch,
              "classifier":trained_model.classifier,
              "activation_layers":"nn.ReLU",
              "state_dict": trained_model.state_dict(),
              "class_to_idx": train_datasets.class_to_idx
             }
    torch.save(checkpoint, train_args.save_dir)
     
  

#call to the main function
if __name__ == "__main__":
    main()
