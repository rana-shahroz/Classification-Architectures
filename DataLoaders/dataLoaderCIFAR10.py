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
    
    # TODO : ALSO write the denormalize function for debugging.
    