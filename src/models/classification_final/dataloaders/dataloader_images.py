import glob, random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import torchvision.transforms.functional as TF
import natsort as ns
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
import torch
import random


class ImageDataset(Dataset):
    """
    Dataset class for image-to-image translation tasks.

    Args:
        inputs (str): Path to input images.
        path_df (str): Path to dataframe with labels.
        transforms (torchvision.transforms.Compose, optional): Transformations to apply to images.
            Defaults to None.
        augmentation (bool, optional): Apply random augmentations to images. Defaults to False.
    """
    
    def __init__(self, path_images:str, path_data:str, transforms: T.Compose = None, general_data:str = ""):
        
        self.transforms     = transforms        
        self.data           = pd.read_csv(path_data)
        self.path_images    = path_images
        self.general_data   = pd.read_csv(general_data) if general_data else None
                
    def __getitem__(self, index):
        """
        Return a pair of input and output images.

        Args:
            index (int): Index of the image pair.

        Returns:
            tuple: Input and output images, and the image name.
        """
        sample      = index % len(self.data)
        
        # Load image
        name_image  = self.data.iloc[sample]["ID"]
        im_input_   = np.load(os.path.join(self.path_images, f"{name_image}"))
        im_input_   = self.transforms(im_input_)
        
        target      = self.data.iloc[sample]["Etiqueta"]
        target      = torch.tensor(target, dtype=torch.long)
        
        data_clinic = self.data.copy().drop(columns=["ID", "Etiqueta"])
        data_clinic = data_clinic.iloc[sample].values
        data_clinic = torch.tensor(data_clinic, dtype=torch.float32)

        return im_input_, data_clinic, target
    
    def __len__(self):
        """
        Return the number of image pairs in the dataset.

        Returns:
            int: Number of image pairs.
        """
        return len(self.data)
    

class Loader:
    """
    Class that creates Data Loaders

    Args:
        images_dir (str): Path to the images directory.
        dat_dir (str): Path to the dat directory.
        augmentation (bool, optional): Apply random augmentations to images. Defaults to False.
    """

    def __init__(self, images_dir: str, data_dir: str,  augmentation: bool = False):
        
        self.images_dir = images_dir

        # Define input and output paths
        self.images_dir = images_dir
        self.train_data = os.path.join(data_dir, "train_clinical_data.csv")
        self.test_data  = os.path.join(data_dir, "test_clinical_data.csv")
        self.val_data   = os.path.join(data_dir, "val_clinical_data.csv")
        self.general_data = os.path.join(data_dir, "parches_con_datos_clinicos.csv")
        
        
        self.transforms_train = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),               # rotaciones leves
            T.RandomVerticalFlip(p=0.2),                # flip vertical (suave)
            RandomGaussianBlur(p=0.3,kernel_size=3,sigma=(0.1, 0.6)),
        ])
        
        self.transforms_test = T.Compose([
            T.ToTensor(),
        ])

        # Create train and test datasets
        self.train_dataset = ImageDataset(
            path_images     = self.images_dir,
            path_data       = self.train_data,
            transforms      = self.transforms_train if augmentation else self.transforms_test,
        )

        self.test_dataset = ImageDataset(
            path_images     = self.images_dir,
            path_data       = self.test_data,
            transforms      = self.transforms_test,
        )
        
        self.val_dataset = ImageDataset(
            path_images     = self.images_dir,
            path_data       = self.val_data,
            transforms      = self.transforms_test,
        )

    def train_dataloader(self, batch_size):
        """
        Create a DataLoader for the training dataset

        Args:
            batch_size (int): Batch size

        Returns:
            DataLoader: Training DataLoader
        """
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def test_dataloader(self, batch_size):
        """
        Create a DataLoader for the test dataset

        Args:
            batch_size (int): Batch size

        Returns:
            DataLoader: Test DataLoader
        """
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    def val_dataloader(self, batch_size):
        """
        Create a DataLoader for the validation dataset

        Args:
            batch_size (int): Batch size

        Returns:
            DataLoader: Validation DataLoader
        """
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)




class RandomGaussianBlur(torch.nn.Module):
    """
    Blur leve para simular pequeñas variaciones del detector.
    """
    def __init__(self, p=0.3, kernel_size=3, sigma=(0.1, 0.8)):
        super().__init__()
        self.p = p
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, img):
        if random.random() < self.p:
            img = self.blur(img)
        return img
if __name__ == "__main__":
    
    images_dir = "/home/kevin-osorno-castillo/Documentos/parches_con_datos_clinicos/imagenes_npy"
    data_dir    = "/home/kevin-osorno-castillo/Documentos/parches_con_datos_clinicos"
    
    # Create an instance of the Loader class
    loader = Loader(images_dir=images_dir, data_dir=data_dir, augmentation=True)
    
    # Create a DataLoader for the datasets
    train_loader    = loader.train_dataloader(batch_size=64)
    test_loader     = loader.test_dataloader(batch_size=2)
    val_loader      = loader.val_dataloader(batch_size=2)
    
    # Print the number of batches in each DataLoader
    print(f"Number of batches in train DataLoader: {len(train_loader)}")
    print(f"Number of batches in test DataLoader: {len(test_loader)}")
    print(f"Number of batches in val DataLoader: {len(val_loader)}")
    
    # Print the shape of the first batch
    for batch in val_loader:
        im_input, data_clinic, target = batch
        #print(f"Input shape: {im_input.shape}")
        print(f"Target shape: {data_clinic.shape}")
    # Print the shape of the first batch
    # for batch in test_loader:
    #     im_input, data_clinic, target = batch
    #     print(f"Input shape: {im_input.shape}")
    #     print(f"Target shape: {target}")
    #     break
    # # Print the shape of the first batch
    # for batch in val_loader:
    #     im_input, data_clinic, target = batch
    #     print(f"Input shape: {im_input.shape}")
    #     print(f"Target shape: {target}")
    #     break