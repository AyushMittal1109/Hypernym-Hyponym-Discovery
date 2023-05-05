# # IMPORTS

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
import copy
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

file = open("/kaggle/input/inlp-project/hypernym-hyponym-dictionaries_2B.pkl",'rb')
parameters = pickle.load(file)
file.close()

vocab = parameters['vocab']
word2index = parameters['w2i']
index2word = parameters['i2w']


print(word2index["bagpipe"])

# # Parameters

# no of projection matrices
k = 24

# no of dimentions in embedding
dim = 300

# no of negative samples
neg_sample_count = 5

# learning rate
lr = 0.001

batch_size = 32

vocab_size = len(vocab)

phase = "training"
subtask = "2B.music"

# datafile
dataFilePath = f"/kaggle/input/inlp-project/{subtask}.{phase}.data.txt"

# goldfile
goldFilePath = f"/kaggle/input/inlp-project/{subtask}.{phase}.gold.txt"

# vocab
vocabFilePath = f"/kaggle/input/inlp-project/{subtask}.vocabulary.txt"


# vocab = []
# with open(vocabFilePath) as dataset:
#     for line in tqdm(dataset):
# #         print(line)
#         line = line.split('\t')
#         vocab.append(line[0][:-1])
# vocab_size = len(vocab)
# print(vocab[:20],vocab_size)

# word2index = {}
# index2word = {}

# word2index['UNK'] = 0
# index2word[0] = 'UNK'
# for i,word in enumerate(vocab):
#     word2index[word] = i+1
#     index2word[i] = word
    
# print(list(word2index.keys())[:20])

# considering preprocesses data like lower and three gram, bi gram, one gram

data = []
with open(dataFilePath) as dataset:
    for line in tqdm(dataset):
        line = line.lower()
        line = line.split('\t')
        data.append(line[0])
        
print(len(data))

gold = []
with open(goldFilePath) as dataset:
    for line in tqdm(dataset):
        line = line.lower()
        line = line.strip()
        line = line.split('\t')
        gold.append(line)
        
print(gold[:20])

def compute_hyponym_hypernyms(data_train_sent,gold_train_sent):
    hyponym_hypernyms = {}
    for i in range(len(data_train_sent)):
        hyponym = data_train_sent[i]
        hypernyms = gold_train_sent[i]
        hyponym_hypernyms[hyponym] = hypernyms
    return hyponym_hypernyms

hyponym_hypernyms = compute_hyponym_hypernyms(data,gold)

# # MODEL ARCHITECTURE

# # loading embeddings

file = open("/kaggle/input/inlp-project/hypernym-hyponym-embeddings_2B.pkl",'rb')
embedding = pickle.load(file)
file.close()

trained_embeddings = torch.randn(dim).to(device)

for word in tqdm(vocab[1:]):
    x = torch.from_numpy(embedding[word]).to(device)
    x = x.reshape([-1])
    trained_embeddings = torch.cat((trained_embeddings,x))
trained_embeddings = trained_embeddings.reshape([-1,dim])
# trained_embeddings = torch.empty((vocab_size, dim),dtype=torch.float32).to(device)
# for i,word in tqdm(enumerate(vocab)):
#     trained_embeddings[i] = torch.from_numpy(embedding[word]).to(device)


class HHD(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(HHD, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(trained_embeddings) #to do

        self.output = nn.Linear(k, 1)
        
        var = 2 / (dim + dim)
        
        # Initialize projection matrices using scheme from Glorot & Bengio (2008).
        
        self.proj_mats = torch.zeros([k, dim, dim], dtype=torch.float32).to(device)
        # Fills self tensor with elements samples from the normal distribution parameterized by mean and std.
        self.proj_mats.normal_(0, var)
        # mat_data is of size k*dim*dim
        # finally mat_data is k*dim*dim matrix ie k projection matrices, each matric is populated with random value
        # diagonal elements will be 1+random value and other will be 0+random value and random value will range 0 and var
        self.proj_mats += torch.cat([torch.eye(dim, ).unsqueeze(0) for _ in range(k)]).to(device)
        self.sigmoid = nn.Sigmoid()
        
#     def similarity(self,query, cand_hypernym):
#         query = self.embedding(query) #1*d
#         cand_hypernymT = self.embedding(cand_hypernym) #1*d
        
#         #proj is of dim d*d, q is 1*d
#         qT = torch.transpose(query,0,1).to(device) # d*1
#         projT = torch.matmul(self.proj_mats,qT).to(device) #k*d*d X d*1 = k*d*1
#         projT = torch.squeeze(projT,2).to(device) #k*d
#         proj = torch.transpose(projT,0,1).to(device) #d*k
        
#         # find similarity between query and candidate 
#         cand_hypernym = torch.transpose(cand_hypernymT,0,1) #d*1
#         simPosHyper = torch.matmul(projT,cand_hypernym).to(device) #k*d x d*1 = k*1
#         simPosHyper = torch.squeeze(simPosHyper,1) # k
#         simPos = self.output(simPosHyper) # 1
        
#         return simPos
    
    def similarity(self,query, cand_hypernym,bs):
        query = self.embedding(query) #1*d
        cand_hypernymT = self.embedding(cand_hypernym) #bs*d
        
        #proj is of dim d*d, q is 1*d
        qT = torch.transpose(query,0,1).to(device) # d*1
        projT = torch.matmul(self.proj_mats,qT).to(device) #k*d*d X d*1 = k*d*1
        projT = torch.squeeze(projT,2).to(device) #k*d
        proj = torch.transpose(projT,0,1).to(device) #d*k
        
        # find similarity between query and candidate 
        cand_hypernym = torch.transpose(cand_hypernymT,0,1) #d*bs
        simPosHyper = torch.matmul(projT,cand_hypernym).to(device) #k*d x d*bs = k*bs
#         simPosHyper = torch.squeeze(simPosHyper,1) # k
        simPosHyper = torch.transpose(simPosHyper,0,1) # bs*k
        simPos = self.output(simPosHyper) # bs*1
        simPos = self.sigmoid(simPos) #bs*1
        
        return simPos
        

    def forward(self, query, cand_hypernym, neg_hypernyms ):
        # query - 255 , cand_hypernym - 255, neg_hypernyms - 255*5
        # getting embeddings of required entities
        query = self.embedding(query) #bs*d
        cand_hypernymT = self.embedding(cand_hypernym) #bs*d
        neg_hypernymsT = self.embedding(neg_hypernyms) #bs*ns*d
        
        query = torch.unsqueeze(query,2) # bs*d*1

#         batch_proj = torch.empty((batch_size,k,dim),dtype=torch.float32).to(device)
#         for i,q in enumerate(query):
#             # q is tensor of size d*1
#             projT = torch.matmul(self.proj_mats,q).to(device) # k*d*d X d*1 = k*d*1
#             projT = torch.squeeze(projT,2) # k*d
#             batch_proj[i] = projT #bs*k*d
            
        batch_proj = torch.tensor([]).to(device)
        for i,q in enumerate(query):
            projT = torch.matmul(self.proj_mats,q).to(device) # k*d*d X d*1 = k*d*1
            projT = torch.squeeze(projT,2) # k*d
            projT = projT.reshape([-1])
            batch_proj = torch.cat((batch_proj,projT))
        
        batch_proj = batch_proj.reshape([-1,k,dim])
        
        
        # find similarity between query and candidate 
        cand_hypernym = torch.unsqueeze(cand_hypernymT,2) #bs*d*1
        simPos = torch.bmm(batch_proj,cand_hypernym) #bs*k*d x bs*d*1 = bs*k*1
        simPos = torch.squeeze(simPos,2) #bs*k
        simPosOutput = self.output(simPos) #bs*1
#         simPosOutput = self.sigmoid(simPosOutput)
        
        
        # a step from above
        # find similarity between query and negative samples
        batch_projT = torch.transpose(batch_proj,1,2) #bs*d*k
        simNegs = torch.bmm(neg_hypernymsT,batch_projT) #bs*ns*d x bs*d*k = bs*ns*k
        simNegsOutput = self.output(simNegs) #bs*ns*1
        simNegsOutput = torch.squeeze(simNegsOutput,2) #bs*ns
#         simNegsOutput = self.sigmoid(simNegsOutput)
        
        
        
        # simPos - bs*1, simNegs - bs*ns
        return simPosOutput,simNegsOutput
    
#     ///////////////////////////////////////////////////////////////////////////////////////
        # getting embeddings of required entities
        query = self.embedding(query)
        cand_hypernymT = self.embedding(cand_hypernym) #1*d
        neg_hypernymsT = self.embedding(neg_hypernyms) #ns*1*d
        
        #proj is of dim d*d, q is 1*d
        qT = torch.transpose(query,0,1) # d*1
        projT = torch.matmul(self.proj_mats,qT).to(device)
        projT = torch.squeeze(projT,2) # k*d*d X d*1 = k*d*1
        proj = torch.transpose(projT,0,1) #k*d
        
        # find similarity between query and candidate 
        cand_hypernym = torch.transpose(cand_hypernymT,0,1) #d*1
        simPosHyper = torch.matmul(proj,cand_hypernym).to(device) #k*d x d*1 = k*1
        simPosHyper = torch.squeeze(simPosHyper,1) # k
        simPos = output(simPosHyper) # 1
        
        # find similarity between query and negative samples
        #neg_hypernyms = torch.transpose(neg_hypernymsT,1,2) # ns*d*1
        simNegHypersT = torch.matmul(neg_hypernymsT,projT).to(device) # ns*1*d x d*k= ns*1*k
        simNegHypers = torch.transpose(simNegHypersT,1,2) # ns*k*1
        simNegHypers = torch.squeeze(simNegHypers,2) # ns*k
        simNegs = output(simNegHypers) # ns*1
        simNegs = torch.squeeze(simNegs,1) # ns
        
        return simPos,simNegs
    
                
        
        


# # Training

# shuffled_vocab = copy.deepcopy(vocab)

def find_negative_samples(x):
    answer = []
    while len(answer)<neg_sample_count:
        hm = random.choice(vocab)
        if hm in hyponym_hypernyms[x] and hm not in vocab:
            continue
        else:
            answer.append(word2index[hm])
    
    return answer

embedding_size = dim
model = HHD(vocab_size,embedding_size)
model.to(device)
# model.to(device)   

criterion = nn.BCEWithLogitsLoss(weight=None, reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(model,data,gold):
    data_size = len(data)
    queries = []
    hypernyms = []
    neg_samples = []
    running_loss = []
    rows = 0


    for i, query in tqdm(enumerate(data)):
        for j, hypernym in enumerate(gold[i]):
            try:
                q = word2index[query]
                h = word2index[hypernym]
            except:
    #             print(query,hypernym)
                continue

            # add query to list
            queries.append(q) #255 - batch size
            # add hypermym to list
            hypernyms.append(h) #255
            # add negative samples to list
            neg_samples.append(find_negative_samples(query)) # 255*5

            rows += 1
            if rows % batch_size == 0:

                # make tensor from query list
                queries_t = torch.tensor(queries, dtype=torch.long).to(device) #255
                
                # make tensor from hypernym list
                hypernyms_t = torch.tensor(hypernyms, dtype=torch.long).to(device) #255
                # make tensor form negative sampleS list
                neg_samples_t = torch.tensor(neg_samples, dtype=torch.long).to(device) #255*5


                # pass to model
                optimizer.zero_grad()
                simPos,simNegs = model(queries_t, hypernyms_t, neg_samples_t) #255*1 , 255*5

    #             output = torch.cat((simPos,simNegs),1) #255*6
    #             print("256*")

                y_pos = torch.ones((simPos.shape[0],1)).to(device) #255*1
                y_neg = torch.zeros((simNegs.shape[0],neg_sample_count)).to(device) #255*5
    #             target = torch.cat((y_pos,y_neg),1) #255*6

                # calculate positive and negative loss
                pos_loss = criterion(simPos, y_pos)
                neg_loss = criterion(simNegs, y_neg)
                loss = neg_loss + pos_loss

                # back propogate the loss
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                # clear the lists

                del queries_t
                del hypernyms_t
                del neg_samples_t

                queries.clear()
                hypernyms.clear()
                neg_samples.clear()

    epoch_loss = np.mean(running_loss)
    print("Training epoch_loss is", epoch_loss)
    return epoch_loss

for i in range(10):
    train(model,data,gold)

def evaluate(model,data,gold):
    
    model.eval()
    
    data_size = len(data)
    queries = []
    hypernyms = []
    neg_samples = []
    running_loss = []
    rows = 0


    for i, query in tqdm(enumerate(data)):
        for j, hypernym in enumerate(gold[i]):
            try:
                q = word2index[query]
                h = word2index[hypernym]
            except:
    #             print(query,hypernym)
                continue

            # add query to list
            queries.append(q) #255 - batch size
            # add hypermym to list
            hypernyms.append(h) #255
            # add negative samples to list
            neg_samples.append(find_negative_samples(query)) # 255*5

            rows += 1
            if rows % batch_size == 0:

                # make tensor from query list
                queries_t = torch.tensor(queries, dtype=torch.long).to(device) #255
                
                # make tensor from hypernym list
                hypernyms_t = torch.tensor(hypernyms, dtype=torch.long).to(device) #255
                # make tensor form negative sampleS list
                neg_samples_t = torch.tensor(neg_samples, dtype=torch.long).to(device) #255*5


                # pass to model
#                 optimizer.zero_grad()
                simPos,simNegs = model(queries_t, hypernyms_t, neg_samples_t) #255*1 , 255*5

    #             output = torch.cat((simPos,simNegs),1) #255*6
    #             print("256*")

                y_pos = torch.ones((simPos.shape[0],1)).to(device) #255*1
                y_neg = torch.zeros((simNegs.shape[0],neg_sample_count)).to(device) #255*5
    #             target = torch.cat((y_pos,y_neg),1) #255*6

                # calculate positive and negative loss
                pos_loss = criterion(simPos, y_pos)
                neg_loss = criterion(simNegs, y_neg)
                loss = neg_loss + pos_loss

                # back propogate the loss
#                 loss.backward()
#                 optimizer.step()
                running_loss.append(loss.item())
                # clear the lists

                del queries_t
                del hypernyms_t
                del neg_samples_t

                queries.clear()
                hypernyms.clear()
                neg_samples.clear()

    epoch_loss = np.mean(running_loss)
    print("Training epoch_loss is", epoch_loss)
    return epoch_loss

evaluate(model,data,gold)

def predict(query):
    data_size = len(data)
    queries = []
    hypernyms = []
    neg_samples = []
    running_loss = []
    rows = 0
    try:
        q = torch.tensor([word2index[query]]).to(device)
    except:
        return "word not found in vocab"
    closest_hypernyms = [] #[[similarity,word]]
#     for i,cand_hypernym in tqdm(enumerate(vocab[:-2])):
#         if query == cand_hypernym:
#             continue
        
#         try:
#             h = torch.tensor([word2index[cand_hypernym]])
#         except:
#             continue
        
#         s = model.similarity(q,h)
        
#         closest_hypernyms.append([s,cand_hypernym])
    
    h = torch.tensor(list(range(1,vocab_size))).to(device)
    s = model.similarity(q,h,h.shape[0]) #bs*1
#     to do - append similarities for all words
    for i in range(1,vocab_size):
        closest_hypernyms.append([float(s[i-1]),vocab[i]])
    closest_hypernyms.sort(reverse=True)
    answer = []
    
    l = 100
    if l>len(closest_hypernyms):
        l = len(closest_hypernyms)
    
    for i in range(l):
        answer.append(closest_hypernyms[i][1])
        
    return answer

predict("tropical_storm")

torch.save(model, '/kaggle/working/HH_Projection_model_2B.pt')

train(model,data,gold)

evaluate(model,data,gold)

for i in range(10):
    train(model,data,gold)
    