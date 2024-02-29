import json
from nltk.util import ngrams
import util
from util import utils
import torch
def get_recall(cand,ref):
    #全部recall计算函数
    recall_list=[]
    for c,r in zip(cand,ref):
        recall_i=recall(c,r)
        recall_list.append(recall_i)
    return recall_list

def recall(c,r):
    count=0
    for word in r:
        if word in c:
            count+=1
    if len(r)!=0:
        recall_value=count/(len(r))
    else:
        recall_value=1
    return recall_value


'''def get_bleu(cand,ref,log_path):
    #全部bleu计算函数
    bleu_list=[]
    for c,r in zip(cand,ref):
        _,bleu_i=utils.eval_bleu([r], [c], log_path)
        #print(bleu_i)
        bleu_list.append(bleu_i)
    return bleu_list'''

def get_bleu(cand,ref,log_path):
    #全部bleu计算函数
    result,bleu=utils.eval_bleu(ref, cand, log_path)
    return bleu

def count_gram(text,n):
    gram_sum=0
    gram_uni=[]
    for sent in text:
        gram=ngrams(sent,n)
        for g in gram:
            gram_sum+=1
            if g not in gram_uni:
                gram_uni.append(g)
    #return len(gram_uni),gram_sum, len(gram_uni)/gram_sum
    return len(gram_uni)/gram_sum

def get_dist(cand):
    gram=[1,2]
    dist=[]
    for n in gram:
        dist_n=count_gram(cand,n)
        dist.append(dist_n)
    return dist[0],dist[1]

def weight_metric(metric,weight,item_num):
    metric=torch.tensor(metric,dtype=float)
    weight=torch.tensor(weight,dtype=float)
    weighted_metric=metric/weight
    weighted_metric=weighted_metric.sum()
    weighted_metric=weighted_metric/item_num
    return weighted_metric
    #对指标加权

def get_weight(item):
    #计算商品在数据集中出现次数
    weight_list=[]
    for i in item:
        weight_list.append(item.count(i))


    return weight_list

def get_max_expose(cand,adv,item,expose):
    cand_new=[]
    adv_new=[]
    num=len(cand)
    item_expose={}
    item_idx={}
    for i in range(num):
        key=str(sorted(item[i]))
        if key not in item_expose:
            item_expose[key]=expose[i]
            item_idx[key]=i
        elif item_expose[key]<expose[i]:
            item_expose[key]=expose[i]
            item_idx[key]=i

    for idx in item_idx.values():
        cand_new.append(cand[idx])
        adv_new.append(adv[idx])

    return cand_new,adv_new
def get_item_num(item):
    item=[str(sorted(i)) for i in item]
    item_num = len(list(set(item)))
    return item_num


def get_metric(cand,adv,item,query,item_query,expose,log_path):
    weight_list=get_weight(item)
    item_num = get_item_num(item)
    print(item_num)

    cand_max_expose, adv_max_expose = get_max_expose(cand, adv, item, expose)
    print(len(cand_max_expose))
    bleu = get_bleu(cand_max_expose, adv_max_expose, log_path)

    recall_q=get_recall(cand,query)
    recall_q=weight_metric(recall_q,weight_list,item_num)

    recall_k=get_recall(cand,item)
    recall_k=weight_metric(recall_k,weight_list,item_num)

    recall_qk=get_recall(cand,item_query)
    recall_qk=weight_metric(recall_qk,weight_list,item_num)

    dist1,dist2=get_dist(cand)

    return bleu,recall_q,recall_k,recall_qk,dist1,dist2