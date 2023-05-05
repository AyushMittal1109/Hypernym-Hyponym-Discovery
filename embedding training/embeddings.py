# %% [markdown]

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import random
import json
import re
from sklearn.manifold import TSNE
from scipy import spatial
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# # Processing data and gold file

# preprocess data and gold file +  convert bi and trigrams to a underscore seperated word

subtask = "2B.music"
phase = "training"

f = open(f'/kaggle/input/inlp-project/{subtask}.{phase}.data.txt','r')
data = f.read()
f.close()
f = open(f'/kaggle/input/inlp-project/{subtask}.{phase}.gold.txt','r')
gold = f.read()
f.close()
f = open(f'/kaggle/input/inlp-project/{subtask}.vocabulary.txt','r')
Vocab = f.read()
f.close()

data = data.split('\n')
gold = gold.split('\n')
Vocab = Vocab.split('\n')

w2i = {}
i2w = {}
vocab = []
ind = 1

w2i['UNK'] = 0
i2w[0] = 'UNK'
vocab.append('UNK')


for line in tqdm(Vocab):
    line = line.lower()
    line = line.split(' ') 
    joined_word = ""
    for w in line:
        joined_word += w +"_"
    joined_word = joined_word[:-1]
    
    w2i[joined_word] = ind
    i2w[ind] = joined_word
    vocab.append(joined_word)
    ind += 1
    
        

hyponyms = []

for line in data:
    line = line.lower()
    line = line.split("\t")
    line = line[0]
    line = line.split(" ")
    if len(line)>1:
        joined_word = ""
        for word in line:
            joined_word += word + "_"
        joined_word = joined_word[:-1]
        if joined_word not in w2i.keys():
            l = len(w2i.keys())
            w2i[joined_word] = l
            i2w[l] = joined_word
            vocab.append(joined_word)
        hyponyms.append(joined_word)
    else:
        hyponyms.append(line[0])
        joined_word = line[0]
        if joined_word not in w2i.keys():
            l = len(w2i.keys())
            w2i[joined_word] = l
            i2w[l] = joined_word
            vocab.append(joined_word)
hyponyms = hyponyms[:-1]

hyponyms[-10:]

hypernyms = []
for line in gold:
    line = line.lower()
    line = line.split("\t")
    temp_hypernyms = []
    for word in line:
        word = word.split(" ")
        if len(word)>1:
            joined_word = ""
            for w in word:
                joined_word += w + "_"
            joined_word = joined_word[:-1]
            if joined_word not in w2i.keys():
                l = len(w2i.keys())
                w2i[joined_word] = l
                i2w[l] = joined_word
                vocab.append(joined_word)
            temp_hypernyms.append(joined_word)
        else:
            temp_hypernyms.append(word[0])
            joined_word = word[0]
            if joined_word not in w2i.keys():
                l = len(w2i.keys())
                w2i[joined_word] = l
                i2w[l] = joined_word
                vocab.append(joined_word)
    hypernyms.append(temp_hypernyms)
    

all_hypernyms = set()

for line in hypernyms:
    for word in line:
        all_hypernyms.add(word)

all_hypernyms = list(all_hypernyms)
# all_hypernyms[:10]

# a function for finding negative hypernyms of given hyponyms
# this function will return hyponym positive and negative hpyernyms in following manner
# given hyponym - 'ayush'
''' function should return - 
[

    [['man'],['neg11','neg12','neg13','neg14','neg15']],
    [['boy'],['neg21','neg22','neg23','neg24','neg25']],
    [['person'],['neg31','neg32','neg33','neg34','neg35']],
    [['student'],['neg41','neg42','neg43','neg44','neg45']],
    
    ]
    
    
    '''

num_neg_hypernyms = 5

def pos_neg_hypernyms(hyponym):

    try:
        index_in_data = hyponyms.index(hyponym)
        
    except:
        print(ind,len(hyponyms),hyponym)
    
    hypernyms_temp = hypernyms[index_in_data]
    num_hypernyms = len(hypernyms_temp)
    neg_hypernyms = []
    for i in range(num_hypernyms*num_neg_hypernyms):
        neg_h = all_hypernyms[random.randint(0,len(all_hypernyms)-1)]
        while neg_h in neg_hypernyms or neg_h == hyponym or neg_h in hypernyms_temp: 
            neg_h = all_hypernyms[random.randint(0,len(all_hypernyms)-1)]
            
        neg_hypernyms.append(neg_h)
    
    ans = []
    for i in range(num_hypernyms):
        ans_temp = []
        h_ind = w2i[hypernyms_temp[i]]
        ans_temp.append([h_ind])
        
                    
        neg_temp = []
        for j in range(i*5,i*5+5):
            neg_temp.append(w2i[neg_hypernyms[j]])
            
        ans_temp.append(neg_temp)
        
        ans.append(ans_temp)
    return ans
        

# # Initilization of embedding from word2vec

# embed_dict = {}

# with open('/kaggle/input/glove-embeddings/glove.6B.300d.txt','r') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:],'float32')
#         embed_dict[word]=vector

# embed_dict['oov'] = np.zeros(300)


f = open('/kaggle/input/word2vec/model.txt','r')
word2vec_pretrained = f.read()
word2vec_pretrained = word2vec_pretrained.split('\n')
word_emb = {}
for i,sent in tqdm(enumerate(word2vec_pretrained)):
    if i == 0 or i == len(word2vec_pretrained)-1:
        continue
    sent = sent.split(' ')
    word_tag = sent[0]
    word_tag = word_tag.split('_')
    word = word_tag[0]
    tag = word_tag[1]
    emb = sent[1:]
    word_emb[word] = emb

my_embed = torch.empty((len(w2i.keys()),300),dtype=torch.float32).to(device)

for i in tqdm(range(len(w2i.keys()))):
    try:
        my_embed[i] = tensor.torch(word_emb[i2w[i]])
    except:
        my_embed[i] = torch.randn(300) - 0.5
#     my_embed.append(x)
    
# my_embed = np.array(my_embed)

print(type(my_embed[0]))
ayush = torch.tensor(my_embed).to(device)

# # Model architecture

class w2v_HH_embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(w2v_HH_embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(my_embed) #to do
        
        self.linear1 = nn.Linear(embedding_size, 1)
        self.linear2 = nn.Linear(embedding_size, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, hyponym, hypernym, neg_hypernym):
        # (bs,1) (bs,1) (bs,neg_sam)
        hyponym_embeddings = self.embedding(hyponym) # bs,1,300
        hypernym_embeddings = self.embedding(hypernym) # bs,1,300
        neg_hypernym_embeddings = self.embedding(neg_hypernym) # bs,5,300
        
#         similarity between hyponym and true hypernym
        pos_score = torch.mul(hyponym_embeddings, hypernym_embeddings) #bs,1,300
        pos_score = torch.squeeze(pos_score, 1)#bs,300  
        pos_score = self.linear1(pos_score)#bs,1        
        pos_score = -F.logsigmoid(pos_score) #bs,1
        pos_score = torch.squeeze(pos_score,1) #bs

#         similarity between hyponym and true neg hypernym
        hyponym_embeddingsT = torch.transpose(hyponym_embeddings, 1, 2) #bs,300, 1
        neg_score = torch.bmm(neg_hypernym_embeddings, hyponym_embeddingsT) #bs,5,1
        neg_score = torch.squeeze(neg_score, 2)#bs,5
        neg_score = -F.logsigmoid(-neg_score) #bs,5
        neg_score = torch.sum(neg_score,dim=1) # bs
        total_score = torch.mean(pos_score + neg_score)
        return total_score

# # Parameters

vocab_size = len(i2w.keys())
epochs = 50
batch_size = 32

model = w2v_HH_embeddings(vocab_size,300)
model.to(device)
optimizer = optim.Adam(model.parameters(),lr=0.0001)

# # Training the embeddings

for epoch in range(epochs):
    hypernym_batch = []
    hyponym_batch = []
    neg_hypernym_batch = []
    running_loss = []
    
    for i,hyponym in tqdm(enumerate(hyponyms)):
#         ind = w2i[hyponyms[i]]
        temp = pos_neg_hypernyms(hyponym)
        
        for a_list in temp:
            # (bs,1) (bs,1) (bs,neg_sam)
            hyponym_batch.append([ind]) #bs*1
            
            hypernym_batch.append(a_list[0]) # bs*1
            neg_hypernym_batch.append(a_list[1]) # bs*5
            
            if len(hyponym_batch) == batch_size:
                a = torch.tensor(hyponym_batch).to(device) # bs*1
                
                b = torch.tensor(hypernym_batch).to(device) # bs*1
                
                c = torch.tensor(neg_hypernym_batch).to(device) # bs*5
                
                loss = model(a,b,c)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                
                hyponym_batch.clear()
                hypernym_batch.clear()
                neg_hypernym_batch.clear()
                
    epoch_loss = np.mean(running_loss)
    print("training epoch_loss is", epoch_loss)
                
#         a = torch.tensor(hyponym_batch)
#         b = torch.tensor(hypernym_batch)
#         c = torch.tensor(neg_hypernym_batch)
#         similarity = model(a,b,c)
''' function should return - 
[

    [['man'],['neg11','neg12','neg13','neg14','neg15']],
    [['boy'],['neg21','neg22','neg23','neg24','neg25']],
    [['person'],['neg31','neg32','neg33','neg34','neg35']],
    [['student'],['neg41','neg42','neg43','neg44','neg45']],
    
    ]
    
    
    '''

torch.save(model, '/kaggle/working/hypernym-hyponym-embeddings_training.pt')

saved_embeddings = {}
for i in tqdm(range(1,len(i2w.keys()))):#(len(index2word)):
    word = i2w[i]
    saved_embeddings[word] = model.embedding.weight[i].detach().cpu().numpy()

with open('hypernym-hyponym-embeddings_2B.pkl','wb') as f:
    pickle.dump(saved_embeddings,f)

parameters = {}

parameters['vocab'] = vocab
parameters['i2w'] = i2w
parameters['w2i'] = w2i
parameters['hypernyms'] = hypernyms
parameters['hyponyms'] = hyponyms

with open('hypernym-hyponym-dictionaries_2B.pkl','wb') as f:
    pickle.dump(parameters,f)