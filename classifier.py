from torch import nn
from torchvision import models
#load the model

def load_model(arch, hidden_units):
    print("Currently loading model {}".format(arch))
    if arch == "vgg16":
        model = models.vgg16(pretrained = True)
        in_features = model.classifier[0].in_features
    elif arch =="densenet121" :  
        model = models.densenet121(pretrained = True)
        in_features = model.classifier.in_features
        print(in_features)
    for params in model.parameters():
        params.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(hidden_units, 2048),
                            nn.ReLU(),
                            nn.Dropout(p = 0.2),
                            nn.Linear(2048, 102),
                            nn.LogSoftmax(dim=1)
                          )

    model.classifier = classifier
    print(model)
    return model