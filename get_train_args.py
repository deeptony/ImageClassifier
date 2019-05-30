#import argparse module
import argparse

#define method to get arguments for training the model
def get_train_args():

    #generating parser object
    parser = argparse.ArgumentParser()
    #adding args to parser
    parser.add_argument("--data_dir", type = str, default = "flowers", help= "Gets file with flower images")
    parser.add_argument("--save_dir", type = str, default = "checkpoint.pt", help= "Set directory to save checkpoint")
    parser.add_argument("--arch", default = "vgg16", help= "Set CNN model")
    parser.add_argument("--learning_rate", type = float, default = 0.001, help= "Set learning rate")
    parser.add_argument("--epochs", type = int, default = 17, help= "Set number of epochs")
    parser.add_argument("--hidden_units", type = int, default = 2048, help= "Set size of input vector for classifier")
    parser.add_argument("--gpu", type = str, default = "", help= "Use GPU ?")

    return parser.parse_args()
