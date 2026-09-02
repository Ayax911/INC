import glob, random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
import torch


class DataClinicDataset(Dataset):
    """
    Dataset class for image-to-image translation tasks.

    Args:
        path_df (str): Path to dataframe with labels.
    """
    
    def __init__(self, path_df:str):
        
        # Read data from CSV
        self.df_target      = pd.read_csv(path_df)
        
        # Create X and y tensors
        X                   = self.df_target.drop(columns=['Diagnostico', 'ID'])
        y                   = self.df_target['Diagnostico']
        self.X              = torch.tensor(X.values, dtype=torch.float32)
        self.y              = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        
        # Return the length of the dataset
        
        return len(self.X)

    def __getitem__(self, idx):
        
        # Get item by index
        return self.X[idx], self.y[idx]
    

class Loader:
    """
    Class that creates Data Loaders

    Args:
        data_dir (str): Path to the dataset directory that contains train, test and val CSV files.
    """

    def __init__(self, data_dir: str):
        
        self.data_dir = data_dir

        # Define input and output paths
        self.train_data    = os.path.join(self.data_dir, "train_clinical_data_processed.csv")
        self.test_data     = os.path.join(self.data_dir, "test_clinical_data_processed.csv")
        self.val_data      = os.path.join(self.data_dir, "val_clinical_data_processed.csv")


        # Create train and test datasets
        self.train_dataset = DataClinicDataset(
            path_df=self.train_data
        )

        self.test_dataset = DataClinicDataset(
            path_df=self.test_data
        )

        self.val_dataset = DataClinicDataset(
            path_df=self.val_data
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
