from tqdm import tqdm

import json
import torch
import re
import os
from transformers import AutoTokenizer#, AutoModel,RobertaForMaskedLM
#tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
tokenizer = AutoTokenizer.from_pretrained("./data")
#model = AutoModel.from_pretrained("ethanyt/guwenbert-large").cuda()
import copy
import numpy as np

from PIL import Image

from torchvision import transforms

data_folder = './real/pic'  # 图片文件夹路径



transform = transforms.ToTensor()
trans_gray=transforms.Grayscale(num_output_channels=1)

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        self.data_all=open(data_path+'restore.txt').readlines()
        self.label = open(data_path + 'true.txt').readlines()
        self.number = len(self.data_all)

        self.folder_path = data_path+'pic'
        self.image_files = [f for f in os.listdir(self.folder_path) if f.endswith('.png')]
        self.image_files = sorted(self.image_files, key=lambda x: (int(re.sub('\D', '', x)), x))

    def __len__(self):
        return  self.number

    def __getitem__(self, index):
        image_name = os.path.join(self.folder_path, self.image_files[index])
        #print(image_name)
        pic = Image.open(image_name)
        pic=transform(pic)[:3,:,:]
        mask_pic=trans_gray(pic)
        assert mask_pic.size(1)==mask_pic.size(2)==64

        data_s= self.data_all[index]
        data_t=self.label[index]
        data_s_token = tokenizer(data_s, padding=True, pad_to_multiple_of=52, return_tensors='pt')
        data_t_token=tokenizer(data_t, return_tensors='pt')['input_ids'][0][1]
        s_ids_mask = data_s_token['input_ids'][0]
        s_mask_pad = data_s_token['attention_mask'][0]
        mask_index=torch.tensor(list(s_ids_mask).index(23291))
        print(image_name)
        print(mask_index)

        example = Example(s_ids_mask,s_mask_pad,mask_pic, mask_index,data_t_token)

        return example

class Collate:
    def __init__(self,ept):
        self.ept=ept

    def __call__(self, example_list):
        return Batch(example_list)

class Batch:
    def __init__(self, example_list):

        self.s_ids_mask=[e.s_ids_mask for e in example_list]
        self.s_mask_pad=[e.s_mask_pad for e in example_list]
        self.mask_pic=[e.mask_pic for e in example_list]
        self.mask_index=[e.mask_index for e in example_list]
        self.tgt_ids=[e.tgt_ids for e in example_list]

class Example:
    def __init__(self,s_ids_mask,s_mask_pad,mask_pic, mask_index,tgt_ids):
        self.s_ids_mask=s_ids_mask
        self.s_mask_pad=s_mask_pad
        self.mask_pic=mask_pic
        self.mask_index=mask_index
        self.tgt_ids=tgt_ids

class DataLoader:

    def __init__(self, data_path,batch_size,data_type):
        self.batch_size=batch_size
        self.dataset = Dataset(data_path=data_path)
        self.data_type=data_type

    def __call__(self):
        assert self.data_type in ['train', 'eval']
        if self.data_type == 'train':
            dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, pin_memory=True,
                                                     collate_fn=Collate(1), shuffle=True)
        else:
            torch.manual_seed(1234)
            dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, pin_memory=True,
                                                     collate_fn=Collate(1), shuffle=False)

        return dataloader


