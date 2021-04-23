import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, BertTokenizer
from typing import Dict, List, Optional, Tuple
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pickle
import random


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Tokenizer:
    def __init__(self, tokenizer: BertTokenizer, max_len) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, max_length=self.max_len , truncation=True, padding="max_length", return_tensors="pt"
        )

    def decode(self, x: Dict[str, torch.LongTensor]):
        return [self.tokenizer.decode(sentence[:sentence_len]) for sentence, sentence_len in 
                zip(x["input_ids"], x["attention_mask"].sum(axis=-1))]


# Dataset Class
class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, file_paths, captions, transform=None, target_tfm=None):
        self.file_paths = file_paths
        self.captions = captions
        self.transform = transform
        self.target_tfm = target_tfm
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        file_path = self.file_paths[idx]
        
        # Read an image with OpenCV
        image = cv2.imread(file_path)
        
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image) 
            image = augmented['image']

        if self.target_tfm:
            tokens = self.target_tfm(caption)

        return image, tokens


# Pytorch-Lightning data model
class LitDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.img_size = args.img_size

        infile = open(os.path.join(self.data_dir, 'filenames_train.pickle'),'rb')
        self.train_indices = pickle.load(infile)
        infile.close()

        infile = open(os.path.join(self.data_dir, 'filenames_test.pickle'),'rb')
        self.valid_indices = pickle.load(infile)
        infile.close()

        self.tokenizer = Tokenizer(AutoTokenizer.from_pretrained(args.text_model), args.max_len)
        self.workers = os.cpu_count()


    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data


    def get_file_caption(self, indices):        
        files = []
        captions = []
        for i in indices:
            files.append(os.path.join(self.data_dir, 'images/') + str(i) + '.jpg')
            text = open(os.path.join(self.data_dir, 'captions/') + str(i) + '.txt')
            caption = text.read()
            caption = caption.split('\n')[:-1]
            captions.append(caption)
            text.close()
        return files, captions


    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]

        self.transform_train = A.Compose([
            A.Resize(self.img_size, self.img_size),            
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()], p=1.)

        self.target_tfm = lambda x: self.tokenizer(random.choice(x))
    
        if stage == 'fit' or stage is None:
            # train dataset
            files, captions = self.get_file_caption(self.train_indices)
            self.dataset_train = AlbumentationsDataset(
                file_paths=files,
                captions=captions,
                transform=self.transform_train,
                target_tfm=self.target_tfm
            )

            # valid dataset
            files, captions = self.get_file_caption(self.valid_indices)
            self.dataset_valid = AlbumentationsDataset(
                file_paths=files,
                captions=captions,
                transform=self.transform_train,
                target_tfm=self.target_tfm
            )


    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, shuffle=True)


    def val_dataloader(self):
        '''returns valid dataloader'''
        return DataLoader(dataset=self.dataset_valid, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, shuffle=False)

