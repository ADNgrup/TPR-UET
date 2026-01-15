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
import math, os
import numpy as np
import torchvision.transforms as T
import nltk
# nltk.download('averaged_perceptron_tagger')

def inject_noisy_correspondence(dataset, noisy_rate, noisy_file =None):
    logger = logging.getLogger("Noisy-dataset")
    nums = len(dataset)
    dataset_copy = dataset.copy()
    captions  = [i[3] for i in dataset_copy]
    images    = [i[2] for i in dataset_copy]
    image_ids = [i[1] for i in dataset_copy]
    pids      = [i[0] for i in dataset_copy]
    len_cap = [len(i.split(" ")) for i in captions]
    print(f"Caption summary : max_len={max(len_cap)} \t| min_len={min(len_cap)} \t| mean_len={np.mean(np.array(len_cap))}")
    # raise "hehe"
    noisy_inx = np.arange(nums)
    if noisy_rate > 0:
        print(noisy_file)
        random.seed(123)
        if os.path.exists(noisy_file):
            logger.info('=> Load noisy index from {}'.format(noisy_file))
            noisy_inx = np.load(noisy_file)
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[0: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            np.save(noisy_file, noisy_inx)

    real_correspondeces = []
    for i in range(nums):
        if noisy_inx[i]== i:
            real_correspondeces.append(1)
        else:
            real_correspondeces.append(0)
        # pid, real_pid, image_id, image_path, text
        tmp = (pids[i],image_ids[i],images[i],captions[noisy_inx[i]])
        dataset[i] = tmp
    logger.info(real_correspondeces[0:10])
    logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(real_correspondeces),nums-np.sum(real_correspondeces), nums))
    return dataset, np.array(real_correspondeces)



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
                 maskT_ratio:float = 0.2,
                 maskA_ratio:float = 0.8,
		         maskG_ratio:float = 0,
                 vision_patch_size:int=16, 
                 noisy_rate=0.0, noisy_file="",
                 datasetname="CUHK-PEDES",**kwargs):
        self.dataset             = dataset
        self.datasetname         = datasetname
        self.transform           = transform
        self.mim_transform       = mim_transform
        self.text_length         = text_length
        self.truncate            = truncate
        self.maskT_ratio         = maskT_ratio
        self.maskA_ratio         = maskA_ratio
        self.maskG_ratio	     = maskG_ratio
        self.noisy_rate          = noisy_rate
        self.noisy_file          = noisy_file
        self.tokenizer           = SimpleTokenizer()
        self.vision_patch_size   = vision_patch_size
        self.repaired_samples    = None; self.check=0
        
        self.dataset, self.real_correspondences = inject_noisy_correspondence(dataset, self.noisy_rate, self.noisy_file)

    def __len__(self):
        if self.repaired_samples is None: return len(self.dataset)
        else: return len(self.repaired_samples)
    
    def selecting_samples(self, index_list):
        if index_list is None: self.repaired_samples = None
        else:
            new_dataset = []
            for sidx in index_list:
                pid, image_id, img_path, caption = self.dataset[sidx]
                new_dataset.append((pid, image_id, img_path, caption))
            self.repaired_samples = new_dataset

    def __getitem__(self, index):
        if self.repaired_samples is None: pid, image_id, img_path, caption = self.dataset[index]
        else: pid, image_id, img_path, caption = self.repaired_samples[index] 

        img_a       = read_image(img_path)
        img_b       = read_image(img_path)
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
            if self.maskG_ratio > 0:
                img_a, _, _ = self._local_grayscale_replacment(img_a, self.maskG_ratio)
                img_b, _, _ = self._local_grayscale_replacment(img_b, self.maskG_ratio)
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        masked_caption_tokens_a = self.txt_data_aug(caption_tokens.clone().cpu().numpy()) 
        masked_caption_tokens_b = self.txt_data_aug(caption_tokens.clone().cpu().numpy()) 
        masked_attribute_tokens_a, mam_a_label = self.masked_attribute_and_labels(caption_tokens.clone().cpu().numpy()) 
        masked_attribute_tokens_b, mam_b_label= self.masked_attribute_and_labels(caption_tokens.clone().cpu().numpy()) 

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images'  : img_a,  
            'images_a': img_a,
            'images_b': img_b,
            'caption_ids': caption_tokens,
            'masked_caption_ids_a':  masked_caption_tokens_a,
            'masked_caption_ids_b':  masked_caption_tokens_b,

            'masked_att_ids_a': masked_attribute_tokens_a,
            'masked_att_ids_b': masked_attribute_tokens_b,
            'masked_att_label_a': mam_a_label,
            'masked_att_label_b': mam_b_label,

            'index':index,
            "none":True
        }
        for k,v in ret.items():
            if isinstance(k, torch.Tensor): print(k, "--->", v.shape)
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

    def _local_grayscale_replacment(self, x, mask_ratio, block_size=16, hard_patch_mask=None, vision_patch_size=16):
        channel, height, width = x.shape
        mask_size_w = width // block_size            #number patch/row
        mask_size_h = height // block_size            #number patch/row
        bw_ratio_h = height // mask_size_h                  #??
        bw_ratio_w = width // mask_size_w                  #??
        len_keep = int(mask_size_w * mask_size_h * (1 - mask_ratio)) #the number of patch will not be masked

        if hard_patch_mask is None:
            noise = torch.rand(mask_size_w * mask_size_h, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise)
            patch_mask = torch.ones([mask_size_h * mask_size_w], device=x.device)
            patch_mask[:len_keep] = 0   #   0 0 0 0 0 0 0 0 0  1 1 1 1 1  1 1 1...
            patch_mask = torch.gather(patch_mask, dim=-1, index=ids_shuffle)  #random mask by ids_restore

        else:
            patch_mask = hard_patch_mask
        patch_mask = patch_mask.reshape(mask_size_h, mask_size_w).long()    #path_mask
        pixel_mask = patch_mask.repeat(bw_ratio_h * bw_ratio_w, 1, 1)  #--> pixel mask of img => it's size = image's size
        pixel_mask = pixel_mask.reshape(bw_ratio_h, bw_ratio_w, mask_size_h, mask_size_w).permute( 2, 0, 3, 1).reshape(1, height, width)

        if block_size > vision_patch_size:
            print("block size > path size --> repeat interleave")
            patch_mask = torch.repeat_interleave(patch_mask, block_size//vision_patch_size, dim=2)
            patch_mask = torch.repeat_interleave(patch_mask, block_size//vision_patch_size, dim=3)

        #rbg tensor --> grayscale tensor
        r, g, b = x.unbind(dim=-3)
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(x.dtype)
        l_img = l_img.unsqueeze(dim=-3)  # rebind channel
        
        new_image = x * (1-pixel_mask) + l_img * pixel_mask
        return new_image, pixel_mask, patch_mask


    def masked_attribute_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        mask_tokens = []
        labels = []
        att_post_tag = ['NN', "NNS", "NNP", "NNPS",
                        "JJ", "JJR", "JJS", 
                         'VB', 'VBN', 'VBG',  
                         'PRP', 'CD']
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                word:str = self.tokenizer.decoder[token]
                tag  = nltk.pos_tag([word.replace("<w/>", "")])[0][1]
                
                # mask token with 15% probability
                if  (tag in att_post_tag) and (not word in ['person', 'Person', 'people', 'he', 'she', 'him']) :
                    prob = random.random()
                    if prob < self.maskA_ratio:
                        prob /= self.maskA_ratio

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            mask_tokens.append(mask)

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            mask_tokens.append(random.choice(token_range))

                        # -> rest 10% randomly keep current token
                        else: mask_tokens.append(token)
                        # append current token to output (we will predict these later)
                        labels.append(token)
                    else:
                        labels.append(0)
                        mask_tokens.append(token)
                    
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
                    mask_tokens.append(token)
            else:
                labels.append(0)
                mask_tokens.append(token)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            mask_tokens[1] = mask

        return torch.tensor(mask_tokens), torch.tensor(labels)
