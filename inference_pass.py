import json
import torch
from torch import nn, optim

def infer(processed_image, model, topk, cat_to_name, gpu):
    #setting current device
    if gpu == "":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    model.to(device)
    #inference pass
    with torch.no_grad():
        model.eval()
        processed_image = processed_image.to(device)
        model = (model.double()).to(device)
        log_ps = model(processed_image.unsqueeze_(0))
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    #inverting class_to_idx dictionary
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    #converting top_p and top_class to arrays
    top_p = top_p.cpu()
    top_class = top_class.cpu()
    top_p = top_p.detach().numpy()
    top_class = top_class.detach().numpy()
    #converting top_class(idx) to classes and then lo tabels
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    classes = []
    labels = []
    for index in top_class[0]:
        classes.append(idx_to_class[index])
    for item in classes:
        labels.append(cat_to_name[item])
        
    return top_p[0], labels
    # TODO: Implement the code to predict the class from an image file