from tqdm import tqdm

import json
import torch
import re
import os
from transformers import AutoTokenizer  # , AutoModel,RobertaForMaskedLM

#tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-large")
tokenizer = AutoTokenizer.from_pretrained("./data")
# model = AutoModel.from_pretrained("ethanyt/guwenbert-large").cuda()
import copy
import math

class Dataset_sp(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_all = json.load(open(data_path))
        self.number = len(self.data_all)

    def __len__(self):
        return self.number

    def __getitem__(self, index):
        data_t, data_s = self.data_all[index]
        data_s_token = tokenizer(data_s, padding=True, pad_to_multiple_of=52, return_tensors='pt')
        ids = data_s_token['input_ids'][0]
        mask_pad = data_s_token['attention_mask'][0]
        s_ids = copy.deepcopy(ids)
        s_mask_pad = copy.deepcopy(mask_pad)

        s_ids_mask, mask_pic, pic, mask_index, tgt_id = get_maskdata_sp(ids, mask_pad, data_t)
        example = Example(s_ids, s_mask_pad, s_ids_mask, mask_pic, pic, mask_index, tgt_id)

        return example


# punc = read_punc()
# sp_mark=tokenizer(punc)['input_ids']
# sp_mark=torch.tensor(sp_mark).transpose(0,1)[1].tolist()
# sp_mark=list(set(sp_mark))
# print(sp_mark)
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
sample_num=1
def get_sample_pic_sp(sample_weight,data_t):
    mask_index = torch.multinomial(sample_weight, sample_num, replacement=False)
    text = data_t[mask_index - 1]
    font = get_font()
    pic_ori, pic_mask = get_pic_sp(text, font)
    return mask_index,pic_ori,pic_mask

def get_maskdata_sp(s_ids,s_mask_pad,data_t):
    sample_weight=get_sample_weight_sp(s_ids,s_mask_pad)
    #print(sample_weight)
    mask_index, pic_ori, pic_mask=get_sample_pic_sp(sample_weight,data_t)
    if pic_ori==None:
        mask_index, pic_ori, pic_mask = get_sample_pic_sp(sample_weight, data_t)
    if pic_ori==None:
        mask_index, pic_ori, pic_mask = get_sample_pic_sp(sample_weight, data_t)


    tgt_id = s_ids[mask_index]
    s_ids[mask_index] = 23291

    return s_ids, pic_mask, pic_ori, mask_index, tgt_id



def get_sample_weight_sp(s_ids, s_mask_pad):
    # idx = s_mask_pad.sum() - 1
    # s_mask_pad[0] = 0
    # s_mask_pad[idx] = 0

    mark = torch.zeros_like(s_ids)
    for m in sp_mark:
        mark += s_ids.eq(m)
    mark = mark.ne(0)
    s_mask_pad = s_mask_pad.float() - mark.float()

    weight_vocab = weight[s_ids.tolist()]
    assert weight_vocab.size() == s_mask_pad.size()
    s_mask_pad = s_mask_pad * weight_vocab
    return s_mask_pad



from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
import random



font_file = r'./纯字体'
path_list = os.listdir(font_file)

def get_font():

    path_to_ttf = random.choice(path_list)
    path_to_ttf = os.path.join(font_file, path_to_ttf)

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
    black = image.eq(0).sum()
    if black <= 510:
        return True


def get_pic_sp(text, font):
    width = 64
    height = 64
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, 'black', font)

    seed = random.randint(1, 10000)

    image = tran1(image)
    if pic_filter(image) == True:
        return None, None
    else:
        image = image.cuda()
        #image_ori = trans_noise(image, seed)
        image_ori = trans_gray(image)
        images = pic_mask_sp(image, width, height)
        image_masks = []
        for image in images:
            if image.ne(0).sum().item() == 0:
                x = random.randint(0, width-1)
                y = random.randint(0, height-1)
                image[:, x, y] = 1
            image_masks.append(trans_noise(image, seed))

        return image_ori, image_masks

#0.5x0.5, 4 pos, 1/4 mask,
def pic_mask_sp1(image, width, height):
    ratio = 0.5
    mask_width = int(ratio * width)
    mask_height = int(ratio * height)

    image1 = copy.deepcopy(image)
    mask_start_x = 0
    mask_start_y = 0
    image1[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 0.0

    image2 = copy.deepcopy(image)
    mask_start_x = mask_width
    mask_start_y = 0
    image2[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 0.0

    image3 = copy.deepcopy(image)
    mask_start_x = 0
    mask_start_y = mask_height
    image3[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 0.0

    image4 = copy.deepcopy(image)
    mask_start_x = mask_width
    mask_start_y = mask_height
    image4[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 0.0

    return [image1, image2, image3, image4]

#0.5x1, 4 pos, 1/2 mask
def pic_mask_sp2(image, width, height):
    half=int(0.5*width)

    mask_width = width
    mask_height = half

    #uper masked
    image1 = copy.deepcopy(image)
    mask_start_x = 0
    mask_start_y = 0
    image1[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 0.0

    #down masked
    image2 = copy.deepcopy(image)
    mask_start_x = 0
    mask_start_y = half
    image2[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 0.0

    mask_width = half
    mask_height = height

    #left masked
    image3 = copy.deepcopy(image)
    mask_start_x = 0
    mask_start_y = 0
    image3[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 0.0

    #right masked
    image4 = copy.deepcopy(image)
    mask_start_x = half
    mask_start_y = 0
    image4[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 0.0

    return [image1, image2, image3, image4]

#0.5x0.5, 4 pos, retained
def pic_mask_sp5(image, width, height):
    ratio = 0.5
    mask_width = int(ratio * width)
    mask_height = int(ratio * height)


    #left, up, retained
    image1 = torch.zeros_like(image)
    mask_start_x = 0
    mask_start_y = 0
    image1[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height]

    #right, up, retained
    image2 =torch.zeros_like(image)
    mask_start_x = mask_width
    mask_start_y = 0
    image2[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]
    #left, down,
    image3 = torch.zeros_like(image)
    mask_start_x = 0
    mask_start_y = mask_height
    image3[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]
    #right, down
    image4 = torch.zeros_like(image)
    mask_start_x = mask_width
    mask_start_y = mask_height
    image4[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    return [image1, image2, image3, image4]
#0.33x0.33, 9 pos, retain
def pic_mask_sp4(image, width, height):
    ratio = 1/3
    mask_width = int(ratio * width)
    mask_height = int(ratio * height)

    image1 = torch.zeros_like(image)
    mask_start_x = 0
    mask_start_y = 0
    image1[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height]

    image2 =torch.zeros_like(image)
    mask_start_x = mask_width
    mask_start_y = 0
    image2[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image3 = torch.zeros_like(image)
    mask_start_x = width-mask_width
    mask_start_y = 0
    image3[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image4 = torch.zeros_like(image)
    mask_start_x = 0
    mask_start_y = mask_height
    image4[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image5 = torch.zeros_like(image)
    mask_start_x = mask_width
    mask_start_y = mask_height
    image5[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height]

    image6 =torch.zeros_like(image)
    mask_start_x = width-mask_width
    mask_start_y = mask_height
    image6[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image7 = torch.zeros_like(image)
    mask_start_x = 0
    mask_start_y = height-mask_height
    image7[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image8 = torch.zeros_like(image)
    mask_start_x = mask_width
    mask_start_y = height-mask_height
    image8[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]
    image9 = torch.zeros_like(image)
    mask_start_x = width-mask_width
    mask_start_y = height-mask_height
    image9[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    return [image1, image2, image3, image4,image5, image6, image7, image8,image9]

#0.5x0.5,9 pos, retain
def pic_mask_sp5(image, width, height):
    ratio = 0.5
    mask_width = int(ratio * width)
    mask_height = int(ratio * height)
    half=int(ratio*width)
    four=int(0.25*width)

    image1 = torch.zeros_like(image)
    mask_start_x = 0
    mask_start_y = 0
    image1[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height]

    image2 =torch.zeros_like(image)
    mask_start_x = four
    mask_start_y = 0
    image2[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image3 = torch.zeros_like(image)
    mask_start_x = half
    mask_start_y = 0
    image3[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image4 = torch.zeros_like(image)
    mask_start_x = 0
    mask_start_y = four
    image4[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image5 = torch.zeros_like(image)
    mask_start_x = four
    mask_start_y = four
    image5[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height]

    image6 =torch.zeros_like(image)
    mask_start_x = half
    mask_start_y = four
    image6[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image7 = torch.zeros_like(image)
    mask_start_x = 0
    mask_start_y = half
    image7[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    image8 = torch.zeros_like(image)
    mask_start_x = four
    mask_start_y = half
    image8[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]
    image9 = torch.zeros_like(image)
    mask_start_x =half
    mask_start_y = half
    image9[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = image[:,
                                                                                                 mask_start_x:mask_start_x + mask_width,
                                                                                                 mask_start_y:mask_start_y + mask_height]

    return [image1, image2, image3, image4,image5, image6, image7, image8,image9]
#0-100 edge
def pic_mask_sp7(image, width, height):
    ratio = [i / 10.0 for i in range(11)]
    #ratio = [i / 10.0 for i in range(3, 11)]
    #ratio=[0.0]
    images1 = []
    images2=[]
    images3=[]
    for r in ratio:

        mask_width = int(r * width)
        mask_height = int(r * height)

        mask_start_x = random.randint(0, width - mask_width)
        mask_start_y = random.randint(0, height - mask_height)


        #white mask
        image_i=copy.deepcopy(image)
        image_i[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 1.0
        images1.append(image_i)

        #black mask
        image_i = copy.deepcopy(image)
        image_i[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 0.0
        images2.append(image_i)

        #remain, black mask
        image_i = torch.zeros_like(image)
        mask_width = width - mask_width
        mask_height = height - mask_height
        mask_start_x = random.randint(0, width - mask_width)
        mask_start_y = random.randint(0, height - mask_height)
        image_i[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = \
            image[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height]
        images3.append(image_i)

    images=images1+images2+images3
    #images=images3
    return images


trans_noise21 = transforms.RandomAffine(degrees=90, fill=0.0)
def pic_mask_sp(image,width,height):
    x=random.random()

    kernel_size = 0
    while kernel_size % 2 != 1:
        # kernel_size = random.randint(2, 10)
        kernel_size = int(torch.randint(2, 30, (1,))[0])
    sigma=int(torch.randint(1, 30, (1,))[0])
    trans_noise1 = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    number = random.randint(0, 20)
    for i in range(number):
        image = pic_mask2(image, width, height,trans_noise1)

    mask_width = random.randint(int(0 * width), width*2)
    mask_height = random.randint(int(0 * height), height*2)
    mask = torch.zeros(3, width + mask_width * 2, height + mask_height * 2, device=torch.device("cuda"))

    mask_start_x = random.randint(0, width +mask_width)
    mask_start_y = random.randint(0, height +mask_height)

    mask[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = -1.0
    mask=trans_noise1(mask)
    mask=trans_noise21(mask)
    mask = mask[:, mask_width:mask_width + width, mask_height:mask_height + height]
    image=image+mask
    image = torch.max(image, torch.tensor(0, device=torch.device("cuda")))

    return [image]

trans_noise22 = transforms.RandomAffine(degrees=90, fill=0.0)


def pic_mask2(image, width, height, trans_noise1):
    x = random.random()
    if x > 1 / 2:
        mask = torch.zeros_like(image)
        mask_width = random.randint(2, int(0.4 * width))
        mask_height = random.randint(2, int(0.4 * height))
        mask_start_x = random.randint(0, width - mask_width)
        mask_start_y = random.randint(0, height - mask_height)
        mask[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = -1.0
        mask = trans_noise1(mask)
        mask = trans_noise22(mask)
        image = image + mask
        image = torch.max(image, torch.tensor(0, device=torch.device("cuda")))
        image = torch.min(image, torch.tensor(1, device=torch.device("cuda")))

    x = random.random()
    if x > 3/4:
        mask = torch.zeros_like(image)
        mask_width = random.randint(2, int(0.2 * width))
        mask_height = random.randint(2, int(0.2 * height))
        mask_start_x = random.randint(0, width - mask_width)
        mask_start_y = random.randint(0, height - mask_height)
        mask[:, mask_start_x:mask_start_x + mask_width, mask_start_y:mask_start_y + mask_height] = 1.0
        mask = trans_noise1(mask)
        mask = trans_noise22(mask)
        image = image + mask
        image = torch.max(image, torch.tensor(0, device=torch.device("cuda")))
        image = torch.min(image, torch.tensor(1, device=torch.device("cuda")))

    return image
class Collate:
    def __init__(self, ept):
        self.ept = ept

    def __call__(self, example_list):
        example_list_filt = []
        for e in example_list:
            if e.pic != None:
                example_list_filt.append(e)

        return Batch(example_list_filt)


class Batch:
    def __init__(self, example_list):
        self.s_ids = [e.s_ids for e in example_list]
        self.s_mask_pad = [e.s_mask_pad for e in example_list]
        self.s_ids_mask = [e.s_ids_mask for e in example_list]
        self.mask_pic = [e.mask_pic for e in example_list]
        self.pic = [e.pic for e in example_list]
        self.mask_index = [e.mask_index for e in example_list]
        self.tgt_ids = [e.tgt_id for e in example_list]


class Example:
    def __init__(self, s_ids, s_mask_pad, s_ids_mask, mask_pic, pic, mask_index, tgt_id):
        self.s_ids = s_ids
        self.s_mask_pad = s_mask_pad
        self.s_ids_mask = s_ids_mask
        self.mask_pic = mask_pic
        self.pic = pic
        self.mask_index = mask_index
        self.tgt_id = tgt_id


class DataLoader_sp:

    def __init__(self, data_path, batch_size, data_type):
        self.batch_size = batch_size
        self.dataset = Dataset_sp(data_path=data_path)
        self.data_type = data_type

    def __call__(self):
        assert self.data_type in ['train', 'eval']
        if self.data_type == 'train':
            dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, pin_memory=True,
                                                     collate_fn=Collate(1), shuffle=True)
        else:
            #torch.manual_seed(1234)
            #torch.manual_seed(1111)
            dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, pin_memory=True,
                                                     collate_fn=Collate(1), shuffle=True)

        return dataloader


