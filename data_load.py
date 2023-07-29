# 用于训练的dataloader
# 不同的模型进行不同的预处理

import torch
import numpy as np
import json
import os
import pickle
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from utils.vocab import Vocabulary

from transformers.models.ofa.tokenization_ofa import OFATokenizer
from transformers import AutoProcessor
from transformers import BertTokenizer

from models.BLIP.blip import init_tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IC_data(Dataset):
    """作为val和test时的dataset"""
    def __init__(self, config, dir, mode):
        super(IC_data, self).__init__()
        self.config = config
        self.data = json.load(open(dir, 'r'))
        self.model = config.model
        # 根据不同的model选择不同的transforms
        self.patch_resize_transform = self.get_transforms(self.model)
        if self.model == 'OFA':
            self.ofa_ckpt = config.ofa_ckpts
            self.tokenizer = OFATokenizer.from_pretrained(self.ofa_ckpt)
        elif self.model == 'BLIP':
            self.tokenizer = init_tokenizer()
        elif self.model == 'GIT':
            self.processor = AutoProcessor.from_pretrained(config.git_distill, local_files_only=True)
            self.tokenizer = self.processor.tokenizer

        self.mode = mode

    def get_transforms(self, model):
        if model == 'OFA':
            self.resolution = 480
            self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            patch_resize_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)])
        elif model == 'BLIP':
            self.resolution = 384
            self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            patch_resize_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)])
        elif model == 'GIT':
            patch_resize_transform = lambda img: self.processor(images=img, return_tensors='pt').pixel_values[0]
        return patch_resize_transform

    def __getitem__(self, item):
        if self.mode == 'train':
            """"""
        else:
            image_path = self.data[item]['filename']
            img = Image.open(image_path)
            patch_img = self.patch_resize_transform(img)
            image_id = self.data[item]['image_id']
            return image_id, patch_img

    def collate_fn_train(self, batch_data):
        """"""

    def collate_fn_eval(self, batch_data):
        image_id, image = zip(*batch_data)
        image = torch.stack(image, dim=0)
        image_feature = {'patch_image': image}
        return image_id, image_feature

    def __len__(self):
        return len(self.data)


class RWConcept_data(Dataset):

    def __init__(self, config, dir, mode):
        super(RWConcept_data, self).__init__()
        self.config = config
        self.data = json.load(open(dir, 'r'))
        self.model = config.model
        # 根据不同的model选择不同的transforms
        self.patch_resize_transform = self.get_transforms(config.model)
        if self.model == 'OFA':
            self.ofa_ckpt = config.ofa_ckpts
            self.tokenizer = OFATokenizer.from_pretrained(self.ofa_ckpt)
        elif self.model == 'BLIP':
            self.tokenizer = init_tokenizer()
        elif self.model == 'GIT':
            self.processor = AutoProcessor.from_pretrained(config.git_distill, local_files_only=True)
            self.tokenizer = self.processor.tokenizer
        self.mode = mode

    def get_transforms(self, model):
        if model == 'OFA':
            self.resolution = 480
            self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            patch_resize_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)])
        elif model == 'BLIP':
            self.resolution = 384
            self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            patch_resize_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)])
        elif model == 'GIT':
            patch_resize_transform = lambda img: self.processor(images=img, return_tensors='pt').pixel_values[0]
        return patch_resize_transform

    def __getitem__(self, item):
        if self.mode == 'train':
            caption = self.data[item]['caption']
            # 不同的模型加载不同的前缀
            if self.model == 'OFA':
                caption = ' '+caption
            elif self.model == "BLIP":
                caption = ' a picture of ' + caption
            elif self.model == 'GIT':
                caption = ' '+caption

            # 不同的模型tokenize的方式不同
            if self.model == 'OFA':
                cap_id = self.tokenizer([caption], return_tensors="pt").input_ids[0]
            elif self.model == 'BLIP':
                text = self.tokenizer(caption, padding='longest', truncation=True, max_length=20, return_tensors="pt")
                cap_id = text.input_ids[0]
                cap_id[0] = self.tokenizer.bos_token_id
            elif self.model == 'GIT':
                cap_id = self.tokenizer([caption], return_tensors="pt").input_ids[0]

            cap_len = cap_id.shape[0]
            if cap_len < self.config.fixed_len:
                if self.model == 'OFA':
                    cap_id = torch.cat([cap_id, torch.ones([self.config.fixed_len-cap_len])], dim=0)
                elif self.model == 'BLIP':
                    cap_id = torch.cat([cap_id, torch.zeros([self.config.fixed_len-cap_len])], dim=0)
                elif self.model == 'GIT':
                    cap_id = torch.cat([cap_id, torch.zeros([self.config.fixed_len-cap_len])], dim=0)
                att_mask = torch.cat([torch.ones([cap_len]), torch.zeros([self.config.fixed_len-cap_len])], dim=0)
            else:
                cap_id = cap_id[:self.config.fixed_len]
                cap_len = self.config.fixed_len
                att_mask = torch.ones(cap_id.shape)

            image_path = self.data[item]['filename']
            img = Image.open(image_path)
            patch_img = self.patch_resize_transform(img)
            label = 0 if self.data[item]['data'] == 'coco' else 1
            return patch_img, cap_id, att_mask, cap_len, label, self.data[item]
        else:
            image_path = self.data[item]['filename']
            img = Image.open(image_path)
            patch_img = self.patch_resize_transform(img)
            image_id = self.data[item]['image_id']
            return image_id, patch_img

    def collate_fn_train(self, batch_data):
        image, cap_id, att_mask, cap_len, label, data_item = zip(*batch_data)
        image = torch.stack(image, dim=0)
        image_feature = {'patch_image': image}
        cap_id = torch.stack(cap_id, dim=0)
        att_mask = torch.stack(att_mask, dim=0)
        cap_len = torch.Tensor(cap_len).int()
        label = torch.Tensor(label).int()
        return image_feature, cap_id.long(), att_mask.long(), cap_len, label, list(data_item)

    def collate_fn_eval(self, batch_data):
        image_id, image = zip(*batch_data)
        image = torch.stack(image, dim=0)
        image_feature = {'patch_image': image}
        return image_id, image_feature

    def __len__(self):
        return len(self.data)


class RWConcept_data_EWC(Dataset):

    def __init__(self, config, dir, mode='train'):
        super(RWConcept_data_EWC, self).__init__()
        self.config = config
        self.data = json.load(open(dir, 'r'))
        self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.resolution = 480
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])
        self.ofa_ckpt = config.ofa_ckpts
        self.tokenizer = OFATokenizer.from_pretrained(self.ofa_ckpt)
        self.mode = mode

    def __getitem__(self, item):
        if self.mode == 'train':
            caption = ' '+self.data[item]['caption']
            cap_id = self.tokenizer([caption], return_tensors="pt").input_ids[0]
            keyword = ' '+self.data[item]['keyword']
            keyword_id = self.tokenizer([keyword], return_tensors="pt").input_ids[0]
            keyword_id = keyword_id[keyword_id > 2]
            cap_len = cap_id.shape[0]
            if cap_len < self. config.fixed_len:
                cap_id = torch.cat([cap_id, torch.ones([self.config.fixed_len-cap_len])], dim=0)
                att_mask = torch.cat([torch.ones([cap_len]), torch.zeros([self.config.fixed_len-cap_len])], dim=0)
            else:
                cap_id = cap_id[:self.config.fixed_len]
                cap_len = self.config.fixed_len
                att_mask = torch.ones(cap_id.shape)
            if_keyword = torch.Tensor([True if (item in keyword_id) else False for item in cap_id])
            image_path = self.data[item]['filename']
            img = Image.open(image_path)
            patch_img = self.patch_resize_transform(img)
            label = 0 if self.data[item]['data'] == 'coco' else 1
            return patch_img, cap_id, att_mask, cap_len, label, if_keyword

    def collate_fn_train(self, batch_data):
        image, cap_id, att_mask, cap_len, label, if_keyword = zip(*batch_data)
        image = torch.stack(image, dim=0)
        image_feature = {'patch_image': image}
        cap_id = torch.stack(cap_id, dim=0)
        if_keyword = torch.stack(if_keyword, dim=0)
        att_mask = torch.stack(att_mask, dim=0)
        cap_len = torch.Tensor(cap_len).int()
        label = torch.Tensor(label).int()
        return image_feature, cap_id.long(), att_mask.long(), cap_len, label, if_keyword

    def __len__(self):
        return len(self.data)


def data_load_rwc_EWC(config, dir, mode):
    dataset = RWConcept_data_EWC(config, dir, mode)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             collate_fn=dataset.collate_fn_train,
                             num_workers=config.num_workers,
                             pin_memory=True,
                             )
    return data_loader


def data_load(config, dir, mode):
    if mode == 'train':
        print("warning: the train_loader is not exist")
    dataset = IC_data(config, dir, mode)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch_size if mode == 'train' else config.val_batch_size,
                             shuffle=True if mode == 'train' else False,
                             collate_fn=dataset.collate_fn_train if mode == 'train' else dataset.collate_fn_eval,
                             num_workers=config.num_workers,
                             pin_memory=True,
                             )
    return data_loader

def data_load_rwc(config, dir, mode):
    dataset = RWConcept_data(config, dir, mode)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch_size if mode == 'train' else config.val_batch_size,
                             shuffle=False,
                             collate_fn=dataset.collate_fn_train if mode == 'train' else dataset.collate_fn_eval,
                             num_workers=config.num_workers,
                             pin_memory=True,
                             )
    return data_loader


