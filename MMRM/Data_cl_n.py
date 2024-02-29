
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
class Dataset(torch.utils.data.Dataset):
    def __init__(self,epoch,data_path):
        self.data_all=json.load(open(data_path))
        self.number = len(self.data_all)
        self.epoch=epoch

    def __len__(self):
        return  self.number

    def __getitem__(self, index):
        data_t,data_s= self.data_all[index]
        data_s_token = tokenizer(data_s, padding=True, pad_to_multiple_of=52, return_tensors='pt')
        ids = data_s_token['input_ids'][0]
        mask_pad = data_s_token['attention_mask'][0]
        s_ids = copy.deepcopy(ids)
        s_mask_pad = copy.deepcopy(mask_pad)

        s_ids_mask, mask_pic,pic, mask_index, tgt_id = get_maskdata(ids, mask_pad, data_t,self.epoch)
        example = Example(s_ids,s_mask_pad,s_ids_mask, mask_pic,pic, mask_index, tgt_id)

        return example

#punc = read_punc()
#sp_mark=tokenizer(punc)['input_ids']
#sp_mark=torch.tensor(sp_mark).transpose(0,1)[1].tolist()
#sp_mark=list(set(sp_mark))
#print(sp_mark)
import math
sp_mark=[0,2,4,5,1543,2494,219]
vocab = open('./data/vocab.txt').read().splitlines()
#print(vocab[:10])
wordcount = {}
for char in vocab:
    wordcount[char] = 0
datas = json.load(open('./data/train.json'))
for d in tqdm(datas):
    d=d[1]
    for c in d:
        if c in wordcount.keys():
            wordcount[c] += 1
weight = []
for c in vocab:
        weight.append(wordcount[c])
#print(weight[:20])
weight[5]=0
thr=[]
for i in weight:
    if i!=0:
        thr.append(i)
thrs=sum(thr)/float(len(thr))

weight=[max([thrs,w])/thrs for w in weight]
weight=[math.sqrt(1/float(w)) for w in weight]
weight=torch.tensor(weight)

#print(weight[2000:2100])
import random

def get_sample_pic(sample_weight,data_t,epoch):
    #sample_num = random.randint(1, 5)
    sample_num=5
    max_sample_num=sample_weight.ne(0).sum()
    sample_num=min(sample_num,max_sample_num)
    mask_index = torch.multinomial(sample_weight, sample_num, replacement=False).tolist()

    mask_index_i=torch.tensor([random.choice(mask_index)])

    text = data_t[mask_index_i - 1]
    font = get_font()
    pic_ori, pic_mask = get_pic(text, font,epoch)
    return mask_index,mask_index_i,pic_ori,pic_mask

def get_maskdata(s_ids,s_mask_pad,data_t,epoch):
    sample_weight=get_sample_weight(s_ids,s_mask_pad)
    #print(sample_weight)
    mask_index,mask_index_i, pic_ori, pic_mask=get_sample_pic(sample_weight,data_t,epoch)
    if pic_ori==None:
        mask_index,mask_index_i, pic_ori, pic_mask = get_sample_pic(sample_weight, data_t,epoch)
    if pic_ori==None:
        mask_index,mask_index_i, pic_ori, pic_mask = get_sample_pic(sample_weight, data_t,epoch)

    tgt_id = s_ids[mask_index_i]

    for i in mask_index:
        s_ids[i] = 23291


    return s_ids, pic_mask, pic_ori, mask_index_i, tgt_id

# def get_maskdata(s_ids, s_mask_pad, data_t):
#     sample_weight = get_sample_weight(s_ids, s_mask_pad)
#     # print(sample_weight)
#     mask_index = torch.multinomial(sample_weight, sample_num, replacement=False)
#     tgt_id = s_ids[mask_index]
#     s_ids[mask_index] = 23291
#     text = data_t[mask_index - 1]
#     font = get_font()
#     pic_ori, pic_mask = get_pic(text, font)
#     return s_ids, pic_mask, pic_ori, mask_index, tgt_id


def get_sample_weight(s_ids,s_mask_pad):
    #idx = s_mask_pad.sum() - 1
    #s_mask_pad[0] = 0
    #s_mask_pad[idx] = 0


    mark=torch.zeros_like(s_ids)
    for m in sp_mark:
        mark+=s_ids.eq(m)
    mark=mark.ne(0)
    s_mask_pad = s_mask_pad.float() - mark.float()
    weight_vocab=weight[s_ids.tolist()]
    assert weight_vocab.size()==s_mask_pad.size()
    s_mask_pad=s_mask_pad*weight_vocab
    return s_mask_pad


from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
import random

font_file = r'./纯字体'
path_list = os.listdir(font_file)
def get_font():
    path_to_ttf = random.choice(path_list)
    path_to_ttf=os.path.join(font_file, path_to_ttf)

    font = ImageFont.truetype(path_to_ttf, 64, encoding="unic")  # 设置字体
    return font

tran1 = transforms.ToTensor()
trans_gray=transforms.Grayscale(num_output_channels=1)
trans_noise2 = transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1), fill=1.0)
trans_noise3 = transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.2, 1))
# trans_noise4 = [transforms.RandomAdjustSharpness(sharpness_factor=0.7)
def trans_noise(image,seed):
    torch.manual_seed(seed)

    #kernel_size = 5
    kernel_size = 0
    while kernel_size % 2 != 1:
        #kernel_size = random.randint(2, 10)
        kernel_size=int(torch.randint(2,10,(1,))[0])

    trans_noise1= transforms.GaussianBlur(kernel_size=kernel_size,sigma=(1,10))
    image=trans_noise2(image)
    image = torch.min(torch.max((torch.randn_like(image) / (1+1* torch.rand(1).cuda()) + image), torch.tensor(0).cuda()),
                      torch.tensor(1).cuda())

    image=trans_noise1(image)
    image=trans_noise3(image)
    #image=trans_noise4(image)
    image = trans_gray(image)

    return image


def pic_filter(image):
    black=image.eq(0).sum()
    #if black<=6000:
    if black <= 510:
        return True


def get_pic(text,font,epoch):
    width=64
    height=64
    image= Image.new('RGB', (width,height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, 'black', font)

    seed = random.randint(1, 10000)

    image= tran1(image)
    if pic_filter(image)==True:
        return None,None
    else:
        image=image.cuda()
        #image_ori=trans_noise(image,seed)
        image_ori = trans_gray(image)
        image=pic_mask(image,width,height,epoch)
        image_mask = trans_noise(image,seed)
        if image_mask.ne(0).sum().item()==0:
            x= random.randint(0, width-1)
            y=random.randint(0, height-1)
            image_mask[:,x,y]=1
        return image_ori, image_mask

trans_noise21 = transforms.RandomAffine(degrees=90, fill=0.0)
def pic_mask(image,width,height,epoch):
    cl_weight = min(epoch, 10) / 10
    #cl_weight=math.sqrt(cl_weight)

    x=random.random()

    kernel_size = 0
    while kernel_size % 2 != 1:
        # kernel_size = random.randint(2, 10)
        kernel_size = int(torch.randint(2, 30, (1,))[0])
    trans_noise1 = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1, 30))

    number = random.randint(0, 20)
    for i in range(number):
        image = pic_mask2(image, width, height,epoch,trans_noise1)

    mask_width = random.randint(int(0 * width), int(width*2*cl_weight))
    mask_height = random.randint(int(0 * height), int(height*2*cl_weight))
    mask=torch.zeros(3,width+mask_width*2,height+mask_height*2,device=torch.device("cuda"))

    mask_start_x = random.randint(0, width +mask_width)
    mask_start_y = random.randint(0, height +mask_height)

    mask[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = -1.0
    mask=trans_noise1(mask)
    mask=trans_noise21(mask)
    mask = mask[:, mask_width:mask_width + width, mask_height:mask_height + height].cuda()
    image=image+mask
    image = torch.max(image, torch.tensor(0, device=torch.device("cuda")))

    return image


trans_noise22 = transforms.RandomAffine(degrees=90, fill=0.0)
def pic_mask2(image,width,height,epoch,trans_noise1):
    x=random.random()
    if x>1/2:
        cl_weight = min(epoch, 10) / 10
        #cl_weight = math.sqrt(cl_weight)

        mask=torch.zeros_like(image)
        mask_width = random.randint(2, max(int(0.4 * width* cl_weight),2))
        mask_height = random.randint(2,max(int(0.4* height* cl_weight),2))
        mask_start_x = random.randint(0, width - mask_width)
        mask_start_y = random.randint(0, height - mask_height)
        mask[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = -1.0
        mask=trans_noise1(mask)
        mask=trans_noise22(mask)
        image=image+mask
        image = torch.max(image, torch.tensor(0, device=torch.device("cuda")))
        image = torch.min(image, torch.tensor(1, device=torch.device("cuda")))

    x = random.random()
    if x>3/4:
        cl_weight = min(epoch, 10) / 10
        #cl_weight = math.sqrt(cl_weight)

        mask=torch.zeros_like(image)
        mask_width = random.randint(2, max(int(0.2 * width* cl_weight),2))
        mask_height = random.randint(2,max(int(0.2* height* cl_weight),2))
        mask_start_x = random.randint(0, width - mask_width)
        mask_start_y = random.randint(0, height - mask_height)
        mask[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 1.0
        mask=trans_noise1(mask)
        mask=trans_noise22(mask)
        image=image+mask
        image = torch.max(image, torch.tensor(0, device=torch.device("cuda")))
        image = torch.min(image, torch.tensor(1, device=torch.device("cuda")))

    return image


class Collate:
    def __init__(self,ept):
        self.ept=ept

    def __call__(self, example_list):
        example_list_filt=[]
        for e in example_list:
            if e.pic!=None:
                example_list_filt.append(e)

        return Batch(example_list_filt)

class Batch:
    def __init__(self, example_list):

        self.s_ids=[e.s_ids for e in example_list]
        self.s_mask_pad=[e.s_mask_pad for e in example_list]
        self.s_ids_mask=[e.s_ids_mask for e in example_list]
        self.mask_pic=[e.mask_pic for e in example_list]
        self.pic=[e.pic for e in example_list]
        self.mask_index=[e.mask_index for e in example_list]
        self.tgt_ids=[e.tgt_id for e in example_list]


class Example:
    def __init__(self,s_ids,s_mask_pad,s_ids_mask, mask_pic,pic, mask_index, tgt_id):
        self.s_ids=s_ids
        self.s_mask_pad=s_mask_pad
        self.s_ids_mask=s_ids_mask
        self.mask_pic=mask_pic
        self.pic=pic
        self.mask_index=mask_index
        self.tgt_id=tgt_id



class DataLoader:

    def __init__(self, data_path,batch_size,data_type,epoch=None):
        self.batch_size=batch_size
        self.data_type=data_type
        self.epoch=epoch
        self.data_path=data_path

    def __call__(self):
        assert self.data_type in ['train', 'eval']
        if self.data_type == 'train':
            dataloader = torch.utils.data.DataLoader(dataset=Dataset(self.epoch,data_path=self.data_path), batch_size=self.batch_size, pin_memory=True,
                                                     collate_fn=Collate(1), shuffle=True)
        else:
            torch.manual_seed(1234)
            dataloader = torch.utils.data.DataLoader(dataset=Dataset(99,data_path=self.data_path), batch_size=self.batch_size, pin_memory=True,
                                                     collate_fn=Collate(1), shuffle=False)

        return dataloader


