import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, we pre-process them before passing them on to the network
# since we are using torchvision models, we need to output images of size at least 224*224.
# We normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.1,
                           saturation=0.1, hue=0),
    # hue is very important in determining bird species
    transforms.RandomRotation(10, expand=False),
    transforms.RandomResizedCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
