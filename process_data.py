
import torch
from torchvision import models, datasets, transforms
#pass train_args.data_dir as data_dir argument
def process_data(data_dir):
    #define paths for train, validation and test data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #define transforms for train, validation and test data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    #configure datasets
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    #configure loaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = True)

    #return loaders

    return trainloader, validationloader, testloader, train_datasets
