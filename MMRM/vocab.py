import json
from tqdm import tqdm
import math
vocab = open('./data/vocab.txt').read().splitlines()
print(vocab[:100])
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

#weight[5]=0
print(weight[:20])

# thrs=1
# weight_sum=sum(weight)
# weight=[-math.log(max(w,thrs)/weight_sum) for w in weight]
# base=weight[0]
# weight=[w/base for w in weight]


thr=[]
for i in weight:
    if i!=0:
        thr.append(i)
thrs=sum(thr)/float(len(thr))
weight=[max([thrs,w])/thrs for w in weight]
weight=[math.sqrt(1/float(w)) for w in weight]

print(weight[:20])
#print(weight[3000:3020])