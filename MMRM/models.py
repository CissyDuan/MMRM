import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from transformers import  AutoModel,RobertaForMaskedLM
#tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-large")
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

#from torchvision import models
from torchvision.models import resnet50,RegNet_X_400MF_Weights
import copy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision.models import resnet50,regnet_x_400mf

vocab_size = RobertaForMaskedLM.from_pretrained(
            "./data").roberta.embeddings.word_embeddings.weight.size(0)


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        # Number of channels in the training images. For color images this is 3
        nc = 1

        # Size of z latent vector (i.e. size of generator input)
        #nz = 100

        # Size of feature maps in generator
        ngf = 64

        self.main = nn.Sequential(
            # input is Z, going into a convolution

            #nn.ConvTranspose2d(nz, ngf * 16, kernel_size=(4, 4), stride=(1, 1), bias=False),
            #nn.BatchNorm2d(ngf * 16),
            #nn.ReLU(True),
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(4,4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2,kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        #self.w1 = nn.Linear(nz* 2, nz)
    #
    #
    # def preprocess(self,memory_mask,memory_pic):
    #     output=self.w1(torch.cat((memory_mask,memory_pic),dim=-1))
    #     return output

    def forward(self, input):

        input=input.unsqueeze(-1).unsqueeze(-1)
        output=self.main(input)
        #print(output.size())
        return output
class Model_multi(nn.Module):
    def __init__(self, config,use_cuda):
        super(Model_multi, self).__init__()
        self.encoder = RobertaForMaskedLM.from_pretrained("./data").roberta
        latent_size = self.encoder.embeddings.word_embeddings.weight.size(1)
        #self.encoder_pic =vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for p in self.parameters():
            p.requires_grad = False

        #self.generator=Generator(nz= latent_size)
        checkpoint = torch.load('./data/resnet50-19c8e357.pth')
        self.encoder_pic = resnet50(weights=None)
        self.encoder_pic.load_state_dict(checkpoint)
        self.encoder_pic.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.out = RobertaForMaskedLM.from_pretrained("./data").lm_head

        self.generator = Generator(nz=latent_size)
        #self.layer_norm = nn.LayerNorm(latent_size)

        num_ftrs = self.encoder_pic.fc.in_features
        self.encoder_pic.fc = nn.Linear(num_ftrs,latent_size )
        nn.init.zeros_(self.encoder_pic.fc.weight)
        self.use_cuda = use_cuda


    def forward(self,batch):

        memory_mask = self.forward_text(batch)

        pic = batch.mask_pic
        pic = torch.stack(pic, dim=0)
        if self.use_cuda:
            pic = pic.cuda()
        memory_pic = self.forward_pic(pic)

        memory_tgt=self.forward_tgt(batch)

        output = self.get_text(memory_mask, memory_pic)
        pic=self.get_pic(memory_tgt,memory_pic)

        tgt_pic = torch.stack(batch.pic, dim=0).cuda()
        pic_loss = self.compute_loss_pic(pic, tgt_pic)
        output = F.softmax(output, -1)

        return output, pic_loss


    def get_pic(self,memory_tgt,memory_pic):
        memory_cross=memory_tgt+memory_pic
        #memory_cross = memory_cross.squeeze()
        output_pic = self.generator(memory_cross)
        return output_pic

    def get_text(self,memory_mask,memory_pic):
        memory_cross = memory_mask + memory_pic
        memory_cross = memory_cross.squeeze()
        output=self.out(memory_cross)
        #output_pic = self.generator(memory_cross)

        return output

    def forward_tgt(self,batch):

        src = batch.s_ids
        src_pad_mask = batch.s_mask_pad
        mask_index = batch.mask_index
        src = torch.stack(src, dim=0)
        src_pad_mask = torch.stack(src_pad_mask, dim=0)
        mask_index = torch.stack(mask_index, dim=0)

        if self.use_cuda:
            src, src_pad_mask, mask_index = src.cuda(), src_pad_mask.cuda(), mask_index.cuda()

        input = {'input_ids': src, 'attention_mask': src_pad_mask}
        memory = self.encoder(**input)[0]
        memory = torch.unbind(memory, dim=0)
        mask_index = torch.unbind(mask_index)
        memory_mask = [m[i] for m, i in zip(memory, mask_index)]
        memory_mask = torch.stack(memory_mask).squeeze()
        return memory_mask

    def forward_text(self,batch):
        src=batch.s_ids_mask
        src_pad_mask=batch.s_mask_pad
        mask_index=batch.mask_index
        src=torch.stack(src,dim=0)
        src_pad_mask=torch.stack(src_pad_mask,dim=0)
        mask_index=torch.stack(mask_index,dim=0)

        if self.use_cuda:
            src,src_pad_mask,mask_index=src.cuda(),src_pad_mask.cuda(),mask_index.cuda()

        input={'input_ids':src,'attention_mask':src_pad_mask}
        memory = self.encoder(**input)[0]
        memory=torch.unbind(memory,dim=0)
        mask_index=torch.unbind(mask_index)
        memory_mask=[m[i] for m,i in zip(memory,mask_index)]
        memory_mask=torch.stack(memory_mask).squeeze()

        return memory_mask

    def forward_pic(self, pic):
        memory_pic = self.encoder_pic(pic)
        #memory_pic=memory_pic.unsqueeze(1)
        return memory_pic


    def decode(self, batch):
        output,_=self.forward(batch)

        return output


    def decode_pic(self,batch):
        topk=5
        pic = batch.mask_pic
        pic = torch.stack(pic, dim=0)
        if self.use_cuda:
            pic = pic.cuda()

        memory_mask = self.forward_text(batch)
        memory_pic = self.forward_pic(pic)
        output = self.get_text(memory_mask, memory_pic)

        output = F.softmax(output, -1)
        output_sort, output_sortidx = torch.sort(output, dim=1, descending=True)
        output_sortidx=output_sortidx[:,:topk]
        output_sortidx=torch.unbind(output_sortidx,dim=1)
        pics=None
        for id in output_sortidx:
            memory_tgt = self.forward_tgt(id)
            pic = self.get_pic(memory_tgt, memory_pic)
            if pics==None:
                pics=pic
            else:
                pics=torch.cat((pics,pic),dim=1)

        return output_sortidx,pics

    def compute_loss_text(self, hidden_outputs, targets):
        assert hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous()#.view(-1, hidden_outputs.size(2))
        targets = targets.contiguous().view(-1)
        loss = F.nll_loss(torch.log(outputs), targets,  reduction='mean')
        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets.data).sum()
        num_total = targets.ne(1).data.sum()
        acc = num_correct.float() / num_total.float()
        return loss, acc

    def compute_loss_pic(self, output_pic, target_pic):
        loss = F.mse_loss(output_pic, target_pic,reduction='mean')
        return loss

# class Model_lmp_mt_n(nn.Module):
#     def __init__(self, config,use_cuda):
#         super(Model_lmp_mt_n, self).__init__()
#         self.encoder = RobertaForMaskedLM.from_pretrained("./data").roberta
#         latent_size = self.encoder.embeddings.word_embeddings.weight.size(1)
#         #self.encoder_pic =vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#         for p in self.parameters():
#             p.requires_grad = False
#
#         #self.generator=Generator(nz= latent_size)
#         checkpoint = torch.load('./data/resnet50-19c8e357.pth')
#         self.encoder_pic = resnet50(weights=None)
#         self.encoder_pic.load_state_dict(checkpoint)
#         self.encoder_pic.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#
#         self.out = RobertaForMaskedLM.from_pretrained("./data").lm_head
#
#         self.generator = Generator(nz=latent_size)
#         #self.layer_norm = nn.LayerNorm(latent_size)
#
#         num_ftrs = self.encoder_pic.fc.in_features
#         self.encoder_pic.fc = nn.Linear(num_ftrs,latent_size )
#         nn.init.zeros_(self.encoder_pic.fc.weight)
#         self.use_cuda = use_cuda
#         self.sample_num=config.sample_num
#
#
#     def forward(self,batch):
#
#         memory_mask = self.forward_text(batch)
#
#         pic = batch.mask_pic
#         pic = [[p[0] for p in pic], [p[1] for p in pic]]
#         pic = [torch.stack(pic[0], dim=0).cuda(), torch.stack(pic[1], dim=0).cuda()]
#
#         memory_pic = self.forward_pic(pic)
#         output, pic_g = self.combine_add(memory_mask, memory_pic)
#
#         tgt_pic=batch.pic
#         tgt_pic=[torch.stack(t) for t in tgt_pic]
#         tgt_pic = torch.stack(tgt_pic, dim=0).cuda()
#         # tgt_pic = [[p[0] for p in tgt_pic], [p[1] for p in tgt_pic]]
#         # tgt_pic = [torch.stack(tgt_pic[0], dim=0).cuda(), torch.stack(tgt_pic[1], dim=0).cuda()]
#
#         pic_loss = self.compute_loss_pic(pic_g, tgt_pic)
#
#         output = F.softmax(output, -1)
#         # pic_loss=pic_loss/float(self.step)
#
#         return output, pic_loss
#
#     def forward_text(self,batch):
#         src=batch.s_ids_mask
#         src_pad_mask=batch.s_mask_pad
#         src=torch.stack(src,dim=0)
#         src_pad_mask=torch.stack(src_pad_mask,dim=0)
#
#         mask_index=batch.mask_index
#         # mask_index = [[m[0] for m in mask_index], [m[1] for m in mask_index]]
#         # mask_index = [torch.stack(mask_index[0], dim=0).unsqueeze(-1).cuda(), torch.stack(mask_index[1], dim=0).unsqueeze(-1).cuda()]
#
#         if self.use_cuda:
#             src,src_pad_mask=src.cuda(),src_pad_mask.cuda()
#
#         input={'input_ids':src,'attention_mask':src_pad_mask}
#         memory = self.encoder(**input)[0]
#         memory = torch.unbind(memory, dim=0)
#         memory_mask = [m[i] for m, i in zip(memory, mask_index)]
#         memory_mask = torch.stack(memory_mask)
#         memory_mask=torch.unbind(memory_mask,dim=1)
#         #print(memory_mask.size())
#
#         return memory_mask
#
#     def forward_pic(self, pic):
#         memory_pic=[]
#         for p in pic:
#             memory_pic.append(self.encoder_pic(p))
#
#         return memory_pic
#
#     def combine_add(self,memory_mask,memory_pic):
#         output, output_pic=[],[]
#         for t,p in zip(memory_mask,memory_pic):
#             # print(t.size())
#             # print(p.size())
#             memory_cross = t + p
#             memory_cross = memory_cross.squeeze()
#             output.append(self.out(memory_cross))
#             output_pic.append(self.generator(memory_cross))
#         output=torch.stack(output,dim=1)
#         output_pic=torch.stack(output_pic,dim=1)
#
#         return output,output_pic
#
#     def decode(self, batch):
#         output,_=self.forward(batch)
#
#         return output
#
#     def compute_loss_text(self, hidden_outputs, targets):
#         assert hidden_outputs.size(0) == targets.size(0)
#         outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))
#         targets = targets.contiguous().view(-1)
#         # print(outputs.size())
#         # print(targets.size())
#         loss = F.nll_loss(torch.log(outputs), targets,  reduction='none')
#         pred = outputs.max(dim=1)[1]
#         num_correct = pred.data.eq(targets.data).sum()
#         num_total = targets.ne(1).data.sum()
#         acc = num_correct.float() / num_total.float()
#         return loss, acc
#
#     def compute_loss_pic(self, output_pic, target_pic):
#         # print(output_pic.size())
#         # print(target_pic.size())
#         loss = F.mse_loss(output_pic, target_pic,reduction='none').squeeze()
#         loss=loss.sum(1).sum(1)/(64*64*self.sample_num)
#         return loss


class Model_lmp_mt(nn.Module):
    def __init__(self, config,use_cuda):
        super(Model_lmp_mt, self).__init__()
        self.encoder = RobertaForMaskedLM.from_pretrained("./data").roberta
        latent_size = self.encoder.embeddings.word_embeddings.weight.size(1)
        #self.encoder_pic =vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for p in self.parameters():
            p.requires_grad = False

        #self.generator=Generator(nz= latent_size)
        checkpoint = torch.load('./data/resnet50-19c8e357.pth')
        self.encoder_pic = resnet50(weights=None)
        self.encoder_pic.load_state_dict(checkpoint)
        self.encoder_pic.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.out = RobertaForMaskedLM.from_pretrained("./data").lm_head

        self.generator = Generator(nz=latent_size)
        #self.layer_norm = nn.LayerNorm(latent_size)

        num_ftrs = self.encoder_pic.fc.in_features
        self.encoder_pic.fc = nn.Linear(num_ftrs,latent_size )
        nn.init.zeros_(self.encoder_pic.fc.weight)
        self.use_cuda = use_cuda


    def forward(self,batch):

        pic = batch.mask_pic
        pic = torch.stack(pic, dim=0)
        if self.use_cuda:
            pic = pic.cuda()

        memory_mask = self.forward_text(batch)
        tgt_pic = torch.stack(batch.pic, dim=0).cuda()
        memory_pic = self.forward_pic(pic)
        output, pic = self.combine_add(memory_mask, memory_pic)
        pic_loss = self.compute_loss_pic(pic, tgt_pic)
        output = F.softmax(output, -1)
        # pic_loss=pic_loss/float(self.step)

        return output, pic_loss

    def forward_text(self,batch):
        src=batch.s_ids_mask
        src_pad_mask=batch.s_mask_pad
        mask_index=batch.mask_index
        src=torch.stack(src,dim=0)
        src_pad_mask=torch.stack(src_pad_mask,dim=0)
        mask_index=torch.stack(mask_index,dim=0)

        if self.use_cuda:
            src,src_pad_mask,mask_index=src.cuda(),src_pad_mask.cuda(),mask_index.cuda()

        input={'input_ids':src,'attention_mask':src_pad_mask}
        memory = self.encoder(**input)[0]
        memory=torch.unbind(memory,dim=0)
        mask_index=torch.unbind(mask_index)
        memory_mask=[m[i] for m,i in zip(memory,mask_index)]
        memory_mask=torch.stack(memory_mask).squeeze()

        return memory_mask

    def forward_pic(self, pic):
        memory_pic = self.encoder_pic(pic)
        #memory_pic=memory_pic.unsqueeze(1)
        return memory_pic

    def combine_add(self,memory_mask,memory_pic):
        memory_cross = memory_mask + memory_pic
       # memory_cross=self.generator.preprocess(memory_mask,memory_pic)
        memory_cross = memory_cross.squeeze()
        output=self.out(memory_cross)
        output_pic = self.generator(memory_cross)

        return output,output_pic

    def decode(self, batch):
        output,_=self.forward(batch)

        return output

    def decode_pic(self,batch):
        pic = batch.mask_pic
        pic = torch.stack(pic, dim=0)
        if self.use_cuda:
            pic = pic.cuda()

        memory_mask = self.forward_text(batch)
        memory_pic = self.forward_pic(pic)
        output, pic = self.combine_add(memory_mask, memory_pic)

        output = F.softmax(output, -1)


        return output,pic

    def compute_loss_text(self, hidden_outputs, targets):
        assert hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous()#.view(-1, hidden_outputs.size(2))
        targets = targets.contiguous().view(-1)
        loss = F.nll_loss(torch.log(outputs), targets,  reduction='mean')
        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets.data).sum()
        num_total = targets.ne(1).data.sum()
        acc = num_correct.float() / num_total.float()
        return loss, acc

    def compute_loss_pic(self, output_pic, target_pic):
        loss = F.mse_loss(output_pic, target_pic,reduction='mean').squeeze()
        loss=loss/(64*64)
        #loss=loss.sum(1).sum(1)/(64*64)
        return loss

class Model_lmp(nn.Module):
    def __init__(self, config,use_cuda):
        super(Model_lmp, self).__init__()
        #self.encoder = RobertaForMaskedLM.from_pretrained("ethanyt/guwenbert-base").roberta
        self.encoder = RobertaForMaskedLM.from_pretrained("./data").roberta
        latent_size = self.encoder.embeddings.word_embeddings.weight.size(1)
        #self.encoder_pic =vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for p in self.parameters():
            p.requires_grad = False

        checkpoint = torch.load('./data/resnet50-19c8e357.pth')
        self.encoder_pic = resnet50(weights=None)
        self.encoder_pic.load_state_dict(checkpoint)
        self.encoder_pic.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.out = RobertaForMaskedLM.from_pretrained("./data").lm_head

        num_ftrs = self.encoder_pic.fc.in_features
        self.encoder_pic.fc = nn.Linear(num_ftrs,latent_size)
        nn.init.zeros_(self.encoder_pic.fc.weight)

        self.use_cuda = use_cuda
    def forward(self,batch):
        memory_mask, memory_pic = self.forward_text(batch), self.forward_pic(batch)
        output = self.combine_add(memory_mask, memory_pic)
        output = F.softmax(output, -1)

        return output

    def forward_text(self,batch):
        src=batch.s_ids_mask
        #print(len(src))
        src_pad_mask=batch.s_mask_pad
        mask_index=batch.mask_index
        src=torch.stack(src,dim=0)
        src_pad_mask=torch.stack(src_pad_mask,dim=0)
        mask_index=torch.stack(mask_index,dim=0)

        if self.use_cuda:
            src,src_pad_mask,mask_index=src.cuda(),src_pad_mask.cuda(),mask_index.cuda()

        input={'input_ids':src,'attention_mask':src_pad_mask}
        memory = self.encoder(**input)[0]
        memory=torch.unbind(memory,dim=0)
        mask_index=torch.unbind(mask_index)
        memory_mask=[m[i] for m,i in zip(memory,mask_index)]
        memory_mask=torch.stack(memory_mask).squeeze()

        return memory_mask

    def forward_pic(self,batch):
        pic=batch.mask_pic
        pic=torch.stack(pic,dim=0)

        if self.use_cuda:
            pic=pic.cuda()

        memory_pic=self.encoder_pic(pic)
        return memory_pic

    def combine_add(self,memory_mask,memory_pic):
        memory_cross=memory_mask+memory_pic
        output=self.out(memory_cross)

        return output


    def decode(self, batch):
        memory_mask, memory_pic = self.forward_text(batch), self.forward_pic(batch)
        output= self.combine_add(memory_mask, memory_pic)

        return output

    def compute_loss_text(self, hidden_outputs, targets):
        assert hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous()#.view(-1, hidden_outputs.size(2))
        targets = targets.contiguous().view(-1)

        loss = F.nll_loss(torch.log(outputs), targets,  reduction='mean')
        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets.data).sum()
        num_total = targets.ne(1).data.sum()
        acc = num_correct.float() / num_total.float()
        return loss, acc




class Model_lm(nn.Module):
    def __init__(self, config,use_cuda):
        super(Model_lm, self).__init__()
        self.encoder = RobertaForMaskedLM.from_pretrained("./data").roberta
        # for p in self.parameters():
        #    p.requires_grad = False
        self.out = RobertaForMaskedLM.from_pretrained("./data").lm_head
        self.use_cuda=use_cuda

        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self,batch):
        src=batch.s_ids_mask
        src_pad_mask=batch.s_mask_pad
        mask_index=batch.mask_index
        #tgt=batch.s_ids
        src=torch.stack(src,dim=0)
        src_pad_mask=torch.stack(src_pad_mask,dim=0)
        mask_index=torch.stack(mask_index,dim=0)

        if self.use_cuda:
            src,src_pad_mask,mask_index=src.cuda(),src_pad_mask.cuda(),mask_index.cuda()

        input={'input_ids':src,'attention_mask':src_pad_mask}
        memory = self.encoder(**input)[0]
        memory=torch.unbind(memory,dim=0)
        mask_index=torch.unbind(mask_index)
        memory_mask=[m[i] for m,i in zip(memory,mask_index)]
        memory=torch.stack(memory_mask).squeeze()

        output=self.out(memory)
        output = F.softmax(output, -1)
        return output

    def decode(self, batch):
        src = batch.s_ids_mask
        src_pad_mask = batch.s_mask_pad
        mask_index = batch.mask_index

        src = torch.stack(src, dim=0)
        src_pad_mask = torch.stack(src_pad_mask, dim=0)
        mask_index = torch.stack(mask_index, dim=0)

        if self.use_cuda:
            src, src_pad_mask, mask_index = src.cuda(), src_pad_mask.cuda(), mask_index.cuda()

        input = {'input_ids': src, 'attention_mask': src_pad_mask}
        memory = self.encoder(**input)[0]
        memory = torch.unbind(memory, dim=0)
        mask_index = torch.unbind(mask_index)
        memory_mask = [m[i] for m, i in zip(memory, mask_index)]
        memory= torch.stack(memory_mask).squeeze()

        output = self.out(memory)
        #output = F.softmax(output, -1)

        return output

    def compute_loss(self, hidden_outputs, targets):
        assert hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous()#.view(-1, hidden_outputs.size(2))
        targets = targets.contiguous().view(-1)
        loss = F.nll_loss(torch.log(outputs), targets,  reduction='mean')
        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets.data).sum()
        num_total = targets.ne(1).data.sum()
        acc = num_correct.float() / num_total.float()

        return loss, acc



class Model_pic(nn.Module):
    def __init__(self, config,use_cuda):
        super(Model_pic, self).__init__()
        #self.encoder_pic = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        #self.encoder_pic = torch.hub.load('./data/resnet50-19c8e357.pth', 'resnet50', pretrained=True)
        vocab_size = RobertaForMaskedLM.from_pretrained(
            "./data").roberta.embeddings.word_embeddings.weight.size(0)
        checkpoint=torch.load('./data/resnet50-19c8e357.pth')
        self.encoder_pic=resnet50(weights=None)

        #checkpoint = torch.load('./data/regnet_x_400mf-adf1edd5.pth')
        #self.encoder_pic = regnet_x_400mf()

        self.encoder_pic.load_state_dict(checkpoint)

        self.encoder_pic.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #self.encoder_pic.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


        #for p in self.parameters():
            #p.requires_grad = False

        num_ftrs = self.encoder_pic.fc.in_features
        num_ftrs_m= 768
        self.encoder_pic.fc = nn.Linear(num_ftrs, num_ftrs_m)
        self.out=nn.Linear(num_ftrs_m,vocab_size)
        #nn.init.zeros_(self.encoder_pic.fc.weight)

        #for p in self.encoder_pic.fc.parameters():
            #p.requires_grad=True
        self.use_cuda=use_cuda


    def forward(self,batch):
        pic=batch.mask_pic
        pic=torch.stack(pic,dim=0)
        if self.use_cuda:
            pic=pic.cuda()
        memory=self.encoder_pic(pic)
        output=self.out(memory)
        output = F.softmax(output, -1)
        return output

    def decode(self, batch):
        pic = batch.mask_pic
        pic = torch.stack(pic, dim=0)
        if self.use_cuda:
            pic = pic.cuda()
        memory = self.encoder_pic(pic)
        output=self.out(memory)

        return output

    def compute_loss(self, hidden_outputs, targets):
        assert hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous()#.view(-1, hidden_outputs.size(2))
        targets = targets.contiguous().view(-1)
        loss = F.nll_loss(torch.log(outputs), targets,  reduction='mean')
        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets.data).sum()
        num_total = targets.ne(1).data.sum()
        acc = num_correct.float() / num_total.float()
        return loss, acc

