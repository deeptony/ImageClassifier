#implement the train loop. Call function in train.py
import torch
import torch.nn as nn
import torch.optim as optim


def train_loop(model, trainloader, validationloader, lr, epochs, gpu):
    
    #move model to appropiate device
    if gpu == "":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    print(device)
    #define criterion and optimizer method
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    
 
    #training loop, together with validation loop
    print("Model is now training in {}".format(device))
    epochs = epochs
    train_losses, validation_losses = [], []
    model.to(device)
    print(next(model.parameters()).is_cuda)
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            running_loss += loss.item()
            loss.backward()

            optimizer.step()
        
        else:
            accuracy = 0
            validation_loss = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validationloader:
                    images, labels = images.to(device), labels.to(device)

                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    validation_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            train_losses.append(running_loss/len(trainloader))
            validation_losses.append(validation_loss/len(validationloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(validationloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))
    
    return model