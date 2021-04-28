import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

from pytorch_lightning import LightningDataModule
from pathlib import Path
from PIL import Image
import glob


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


# Pytorch-Lightning data model
class LitDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_data = args.train_data
        self.val_data = args.val_data
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.workers = os.cpu_count()


    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data


    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
    
        if stage == 'fit' or stage is None:

            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor()
            ])
            self.train_data = torchvision.datasets.ImageFolder(root=self.train_data, transform=transform)
            self.val_data = torchvision.datasets.ImageFolder(root=self.val_data, transform=transform)


    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, shuffle=True)


    def val_dataloader(self):
        '''returns val dataloader'''
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, shuffle=False) 
