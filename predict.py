from get_predict_args import get_predict_args
from process_image import process_image
from inference_pass import infer
from load_predictor import load_predictor

def main():
    #get args for prediction 
    predict_args = get_predict_args()
    #prepare image for prediction. Returns processed tensor
    processed_image = process_image(predict_args.image_path)
    print(processed_image.dtype)
    #loads model and moves it to current device
    model = load_predictor(predict_args.checkpoint, predict_args.gpu)  
    #Call inference_pass in inference_pass.py and pass --top_k, --category_names and --GPU
    top_p, classif = infer(processed_image, model, predict_args.top_k, predict_args.cat_to_name, predict_args.gpu)
    print(top_p)
    print(classif)
    
if __name__ == "__main__":
    main()    