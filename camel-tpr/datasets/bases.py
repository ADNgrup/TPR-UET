from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils import read_image
from utils.simple_tokenizer import SimpleTokenizer
# from model.tokenization_bert import BertTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
import math
import numpy as np
import torchvision.transforms as T
import nltk


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("DAPROJECT.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        print('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result




class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform


    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 mim_transform=None, 
                 text_length: int = 77,
                 truncate: bool = True,
                 text_aug: bool = True,
                 maskT_ratio:float = 0.2,
                 vision_patch_size:int=16, **kwargs):
        self.dataset             = dataset
        self.transform           = transform
        self.mim_transform       = mim_transform
        self.text_length         = text_length
        self.truncate            = truncate
        self.txt_aug             = text_aug
        self.maskT_ratio         = maskT_ratio
        self.tokenizer           = SimpleTokenizer()
        self.vision_patch_size   = vision_patch_size

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img_a       = read_image(img_path)
        img_b       = read_image(img_path)

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        # if self.mim_transform is not None: 
            img_mim_a = self.mim_transform(img_mim_a) #try to use image_a instead?
            img_mim_b = self.mim_transform(img_mim_b)
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        masked_caption_tokens_a = self.txt_data_aug(caption_tokens.clone().cpu().numpy()) 
        masked_caption_tokens_b = self.txt_data_aug(caption_tokens.clone().cpu().numpy()) 

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images'  : img_a,  
            'images_a': img_a,
            'images_b': img_b,
            'caption_ids': caption_tokens,
            'masked_caption_ids_a': masked_caption_tokens_a,
            'masked_caption_ids_b':  masked_caption_tokens_b,

            'index':index,
            "none":True
        }
        return ret

    def txt_data_aug(self, tokens):
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        new_tokens = np.zeros_like(tokens)
        aug_tokens = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                if prob < self.maskT_ratio:
                    prob /= self.maskT_ratio
                    # 50% randomly change token to mask token
                    if prob < 0.5:
                        aug_tokens.append(mask) 
                    # 20% randomly change token to random token
                    elif prob < 0.7:
                        aug_tokens.append(random.choice(token_range)) # -> rest 10% randomly keep current token
                    else:
                        None # # 30% randomly remove
                else:
                    # no masking token (will be ignored by loss function later)
                    aug_tokens.append(tokens[i])
            else:
                aug_tokens.append(tokens[i])
        new_tokens[0:len(aug_tokens)] = np.array(aug_tokens)
        return torch.tensor(new_tokens)