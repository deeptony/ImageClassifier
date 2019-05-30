#import argparse module
import argparse

#define method to get arguments for training the model
def get_predict_args():

    #generating parser object
    parser = argparse.ArgumentParser()
    #adding args to parser
    parser.add_argument("--image_path", type = str, default = "flowers/test/29/image_04083.jpg", help= "Enter full image path")
    parser.add_argument("--cat_to_name", type = str, default = "cat_to_name.json", help= "Enter path of json file to convert classes into labels")
    parser.add_argument("--top_k", type = float, default = 3, help= "Set top_k with an int")
    parser.add_argument("--checkpoint", type = str, default = "checkpoint.pt", help= "Enter directory to get model checkpoint")
    parser.add_argument("--gpu", type = str, default = "", help= "Use GPU ?")

    return parser.parse_args()
