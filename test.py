#test the model
import torch
from torch import nn

def test(model, testloader, gpu):
    #setting current device
    if gpu == "":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        
    criterion = nn.NLLLoss()
    accuracy = 0
    test_loss = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            log_ps = model(images)
            test_loss += criterion(log_ps, labels).item()
        
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(("Test Accuracy: {:.3f}".format(accuracy/len(testloader))), 
        ("Test Loss: {:.3f}".format(test_loss/len(testloader)))) 