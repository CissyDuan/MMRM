from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.nn as nn
from optims import Optim
import util
from util import utils
import lr_scheduler as L
import models
from tqdm import tqdm
import sys
import os
import Data
import Data_cl
import Data_sp
import Data_real
import Data_cl_n
from Eval import *
import random
#import torch.optim as optim
from transformers import AutoTokenizer#, AutoModel,RobertaForMaskedLM
#tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-large")
tokenizer = AutoTokenizer.from_pretrained("./data")
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)
from util.nlp_utils import *
import json

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import copy
import math
# config
def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    parser.add_argument('-model', default='lmp_mt', type=str,
                        choices=['lmp_mt','lmp','lm','pic','lmp_multi'])
    parser.add_argument('-cl', default='true', choices=['true','false'],
                        help="cl")
    parser.add_argument('-n', default='false', choices=['true', 'false'],
                        help="miss number")
    parser.add_argument('-gpus', default=[1], type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore',
                        type=str, default='',
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=4321,
                        help="Random seed")
    parser.add_argument('-type', default='train', choices=['train', 'eval','eval_sp','eval_pic','eval_real','eval_multi','eval_real_lm'],
                        help='train type or eval')
    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-length', default='64', type=str,
                        help="shi length")

    opt = parser.parse_args()
    # 用config.data来得到config中的data选项
    config = util.utils.read_config(opt.config)
    return opt, config


# set opt and config as global variables
args, config = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


# Training settings

def set_up_logging():
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging = utils.logging(log_path + 'log.txt')  # 往这个文件里写记录
    logging_csv = utils.logging_csv(log_path + 'record.csv')  # 往这个文件里写记录
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_csv, log_path


logging, logging_csv, log_path = set_up_logging()
use_cuda = torch.cuda.is_available()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
def train(model,dataloader_dev,scheduler,optim,updates):
    max_acc = 0.
    for epoch in range(1, config.epoch + 1):
        if args.cl=='true':
            if args.n=='true':
                dataloader_train = get_dataloader_cl_n(split='train', epoch=epoch + config.epoch_ckp)
            else:
                dataloader_train = get_dataloader_cl(split='train', epoch=epoch + config.epoch_ckp)
        elif args.cl=='false':
            dataloader_train = get_dataloader(split='train')

        total_acc = 0.
        total_loss=0
        total_loss_text=0.
        total_loss_pic = 0.
        start_time = time.time()
        updates_epoch=0

        model.train()

        if epoch == 1:
            optim.set_parameters(model.parameters())
            if config.schedule:
                scheduler = L.SetLR(optim.optimizer)

        scheduler.step()
        logging("Decaying learning rate to %g\n" % scheduler.get_lr()[0])

        for batch in tqdm(dataloader_train):
            model.zero_grad()
            if len(batch.tgt_ids) != 0:
                # print(len(batch.tgt_ids))
                tgt = torch.stack(batch.tgt_ids, dim=0).cuda()

                if args.model == 'lmp':
                    output = model.forward(batch)
                    text_loss, acc = model.compute_loss_text(output, tgt)
                    loss=text_loss
                    pic_loss = torch.tensor(0, device='cuda')

                elif args.model =='lmp_mt':
                    output, pic_loss = model.forward(batch)
                    text_loss, acc = model.compute_loss_text(output, tgt)
                    #print(text_loss.size())
                    #print(pic_loss.size())

                    loss = text_loss + 100 * pic_loss
                    #print(text_loss)
                    #print(100 * pic_loss)

                elif args.model == 'lmp_multi':
                    output, pic_loss = model.forward(batch)
                    text_loss, acc = model.compute_loss_text(output, tgt)
                    loss=text_loss+100*pic_loss

                elif args.model in ['lm', 'pic']:
                    output = model(batch)
                    text_loss, acc = model.compute_loss(output, tgt)
                    loss=text_loss
                    #print(loss)
                    pic_loss=torch.tensor(0,device='cuda')

                if torch.isnan(loss):
                    print('err')
                    print(len(batch.tgt_ids))
                    print(tgt.size())
                    assert 1 == 0
                loss.backward()
                optim.step()
                updates += 1  # 进行了一次更新
                updates_epoch += 1
                total_loss_text += text_loss.data.item()
                total_loss_pic += pic_loss.data.item()
                total_loss += loss.data.item()
                total_acc += acc
                # print(acc)

                #score = eval(model, dataloader_dev, epoch, updates,test=False)
                # save_model(log_path + str(updates) + '_updates_checkpoint.pt', model, optim, updates)

                if updates_epoch == 100:
                    logging(
                        "time: %6.3f, epoch: %3d, updates: %8d, train loss text: %6.3f, train loss pic: %6.3f, train loss: %6.3f, train acc: %.3f\n"
                        % (time.time() - start_time, epoch, updates, total_loss_text / updates_epoch,
                           total_loss_pic / updates_epoch, total_loss / updates_epoch,
                           total_acc / updates_epoch))
                    total_acc = 0.
                    total_loss_text = 0.
                    total_loss_pic = 0.
                    total_loss = 0.
                    updates_epoch = 0

                    # acc=eval(model, dataloader_dev, epoch, updates)
                    # if acc >= max_acc:
                    #     save_model(log_path + str(acc) + '_checkpoint.pt', model, optim, updates)
                    #     max_acc = acc

        logging("learning rate to %g" % scheduler.get_lr()[0])
        # print('evaluating after %d updates...\r' % updates)
        # TODO: fix eval and print bleu, ppl
        torch.cuda.empty_cache()
        acc = eval(model, dataloader_dev, epoch, updates)
        # acc=acc.data.item()
        if acc >= max_acc:
            save_model(log_path + str(acc) + '_checkpoint.pt', model, optim, updates)
            max_acc = acc
        # model.train()
    save_model(log_path + str(updates) + '_updates_checkpoint.pt', model, optim, updates)
    if args.n=='true':
        dataloader_test = get_dataloader_cl_n(split='test')
    else:
        dataloader_test = get_dataloader(split='test')
    torch.cuda.empty_cache()
    eval(model, dataloader_test, 0, updates)

    if args.model != 'lm':
        if args.n=='true':
            dataloader_test = get_dataloader_cl_n(split='test')
            eval_multi_n(model, dataloader_test)
        else:
            dataloader_test = get_dataloader_sp(split='test')
            eval_multi(model, dataloader_test)
    if args.model == 'lmp_mt':
        dataloader_real=get_dataloader_real(split='test')
        eval_real(model, dataloader_real)
    return max_acc


from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
tran1 = transforms.ToTensor()
trans_pil=transforms.ToPILImage()


def list_of_groups(init_list, childern_list_len):
    list_of_group = zip(*(iter(init_list),) * childern_list_len)  # zip(childern_list_len ge list_iterator object)
    end_list = [list(i) for i in list_of_group]  # i is a tuple
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

from torch.nn.utils.rnn import pad_sequence

def image_combine(image_show, num):
    image_show = list_of_groups(image_show, num)
    image_show = [torch.cat(img, dim=1).transpose(0, 1) for img in image_show]

    image_show = pad_sequence(image_show, batch_first=True, padding_value=1.0).transpose(1, 2).unbind()
    image_show = torch.cat(image_show, dim=2)
    return image_show


def save_model(path, model, optim, updates):

    checkpoints = {
        'model': model.state_dict(),
        'config': config,
        'updates': updates, 'optim': optim}

    torch.save(checkpoints, path)

def get_dataloader(split):
    assert split in ['train','dev','test']
    if split=='train':
        dataloader = Data.DataLoader(config.data + split+'.json', config.batch_size,data_type='train')
    else:
        dataloader = Data.DataLoader(config.data + split +'.json',int((config.batch_size)),data_type='eval')
    return dataloader()

def get_dataloader_cl(split,epoch=None):
    assert split in ['train','dev','test']
    if split=='train':
        dataloader = Data_cl.DataLoader(config.data + split+'.json', config.batch_size,data_type='train',epoch=epoch)
    else:
        dataloader = Data_cl.DataLoader(config.data + split +'.json',int((config.batch_size)),data_type='eval')
    return dataloader()

def get_dataloader_sp(split):
    assert split in ['test']
    dataloader = Data_sp.DataLoader_sp(config.data + split +'.json', int((config.batch_size)),data_type='eval')
    return dataloader()

def get_dataloader_real(split):
    assert split in ['test']
    dataloader = Data_real.DataLoader('./real/', int((config.batch_size)),data_type='eval')
    return dataloader()

def get_dataloader_cl_n(split,epoch=None):
    assert split in ['train','dev','test']
    if split=='train':
        dataloader = Data_cl_n.DataLoader(config.data + split+'.json', config.batch_size,data_type='train',epoch=epoch)
    else:
        dataloader = Data_cl_n.DataLoader(config.data + split +'.json',int((config.batch_size)),data_type='eval')
    return dataloader()

def main():
    # 设定种子
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    # model
    print('building model...\n')

    if args.model =='lmp_mt':
        model=models.Model_lmp_mt(config, use_cuda)
    elif args.model =='lmp':
        model=models.Model_lmp(config, use_cuda)
    elif args.model == 'lm':
        model = models.Model_lm(config, use_cuda)
    elif args.model=='pic':
        model=models.Model_pic(config, use_cuda)
        print('load pic')
    elif args.model == 'lmp_multi':
        model = models.Model_multi(config, use_cuda)



    if args.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(os.path.join(log_path, args.restore))
        model.load_state_dict(checkpoints['model'])
        print('load model')

    else:

        if args.model in ['lmp', 'lmp_mt','lmp_multi']:
            model_lm = models.Model_lm(config, use_cuda)
            if args.n=='true':
                checkpoints_lm = torch.load(os.path.join(config.lm_path_n, config.lm_restore_n))
            else:
                checkpoints_lm = torch.load(os.path.join(config.lm_path, config.lm_restore))
            model_lm.load_state_dict(checkpoints_lm['model'])
            model.encoder.load_state_dict(model_lm.encoder.state_dict())
            model.out.load_state_dict(model_lm.out.state_dict())
            print('load lm')

            for p in model.encoder.parameters():
                p.requires_grad = False

            # model_pic = models.Model_pic(config, use_cuda)
            # checkpoints_pic= torch.load(os.path.join(config.pic_path, config.pic_restore))
            # model_pic.load_state_dict(checkpoints_pic['model'])
            # model.encoder_pic.load_state_dict(model_pic.encoder_pic.state_dict())
            # print('load pic')

    if use_cuda:
        model.cuda()

    param_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #param_pretrain= sum(p.numel() for p in model.encoder.parameters() )
    param_count=sum(param.numel() for param in model.parameters())
    param_pretrain=param_count-param_train

    logging('total number of parameters: %d\n\n' % param_count)
    logging('total number of pretrain parameters: %d\n\n' % param_pretrain)
    logging('total number of train parameters: %d\n\n' % param_train)

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # updates是已经进行了几个epoch, 防止中间出现程序中断的情况.
    if args.restore:
        updates = checkpoints['updates']
    else:
        updates = 0

    #optimizer = optim.Adam(self.params, lr=self.lr)
    optim= Optim(config.optim, config.learning_rate, config.max_grad_norm,
                         lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)

    if args.type=='train':

        start_time = time.time()
        if args.n=='true':
            dataloader_dev=get_dataloader_cl_n(split='dev',)
        else:
            dataloader_dev = get_dataloader(split='dev')
        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))

        max_acc = train(model,  dataloader_dev,None,optim,updates)
        logging("Best acc score: %.2f\n" % (max_acc))

    elif args.type == 'eval':
        # Load data
        start_time = time.time()
        if args.n=='true':
            dataloader_test = get_dataloader_cl_n(split='test')
        else:
            dataloader_test = get_dataloader(split='test')


        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))
        if args.model!='lm':
            assert args.restore is not None
        eval( model,dataloader_test, 0, updates)

    elif args.type == 'eval_sp':
        # Load data
        start_time = time.time()
        dataloader_test = get_dataloader_sp(split='test')

        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))
        assert args.restore is not None
        eval_sp( model,dataloader_test)
    elif args.type == 'eval_pic':
        # Load data
        start_time = time.time()
        dataloader_test = get_dataloader(split='test')

        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))
        assert args.restore is not None
        eval_pic(model, dataloader_test)
    elif args.type == 'eval_multi':
        # Load data
        start_time = time.time()
        if args.n=='true':
            dataloader_test = get_dataloader_cl_n(split='test')
        else:
            dataloader_test = get_dataloader_sp(split='test')

        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))
        assert args.restore is not None
        if args.n=='true':
            eval_multi_n(model, dataloader_test)
        else:
            eval_multi(model, dataloader_test)
    elif args.type == 'eval_real':
        # Load data
        start_time = time.time()
        dataloader_test = get_dataloader_real(split='test')

        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))
        assert args.restore is not None
        eval_real(model, dataloader_test)
    elif args.type == 'eval_real_lm':
        # Load data
        start_time = time.time()
        dataloader_test = get_dataloader_real(split='test')

        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))
        assert args.restore is not None
        eval_real_lm(model, dataloader_test)
    else:
        print('error')



if __name__ == '__main__':
    main()
