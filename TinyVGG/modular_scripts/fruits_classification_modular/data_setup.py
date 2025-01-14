"""
Contains functionality for creating a PyTorch DataLoaders for image classification task

"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = 0

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       train_image_transform: transforms.Compose,
                       test_image_transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int=NUM_WORKERS):
  """
  Takes in the training directory and testing directory path and turns them
  into a PyTorch Datsets and then into a PyTorch DataLoaders

  Args:
    train_dir: path to training directory
    test_dir: path to testing directory
    transform: torchvision transforms to perform on training and testing data
    batch_size: number of samples per batch in each of the DataLoader
    num_workers: an integer for number of workers per DataLoader
  
  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names)
  """
  
  # Using ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(root=train_dir,
                                    transform=train_image_transform,
                                    target_transform=None)
  test_data = datasets.ImageFolder(root=test_dir,
                                  transform=test_image_transform)
  
  # Get class names
  class_names = train_data.classes
  
  # Turn train and test datsets into DataLoaders
  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True)

  test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)
  
  return train_dataloader, test_dataloader, class_names
