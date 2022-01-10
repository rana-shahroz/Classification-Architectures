# Importing libraries that are needed
import torch
from torchvision import datasets
from torchvision import transforms

def dataLoadCIFAR10(args):
    
    # Transform to be applied to the dataset.
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
    ])
        
    # Downloading the train and test sets.
    dataTrain = datasets.CIFAR10(root = args.path_data,
                                 train = True, 
                                 download = True,
                                 transform=transform
                                 )
    
    
    dataTest = datasets.CIFAR10(root = args.path_data, 
                                train = False,
                                download = True,
                                transform = transform
                                )
    
    # Making the loaders for train and test data.
    trainLoader = torch.utils.data.DataLoader(
        dataset = dataTrain,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        drop_last = True,
        timeout = 0,
        persistent_workers = False
    )
    
    testLoader = torch.utils.data.DataLoader(
        dataset = dataTest,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        drop_last = True,
        timeout = 0,
        persistent_workers = False
    )
    
    return trainLoader, testLoader
    