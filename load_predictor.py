import torch

def load_predictor(path, gpu):
    if gpu == "":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model = checkpoint["model"]
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint["class_to_idx"]
    model = model.to(device)
    return model