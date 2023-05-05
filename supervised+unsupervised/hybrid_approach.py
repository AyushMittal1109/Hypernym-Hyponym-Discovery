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
import copy
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

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

# 1A 2A 2B
subtask = "1A"

# training test
phase = "training"

# datafile
dataFilePath = f"/kaggle/input/inlp-project/{subtask}.english.{phase}.data.txt"

# goldfile
goldFilePath = f"/kaggle/input/inlp-project/{subtask}.english.{phase}.gold.txt"

# vocab
vocabFilePath = f"/kaggle/input/inlp-project/{subtask}.english.vocabulary.txt"

# # Data loading and preprocessing

file = open(f"/kaggle/input/inlp-project/hypernym-hyponym-dictionaries_{subtask}.pkl",'rb')
parameters = pickle.load(file)
file.close()

vocab = parameters['vocab']
w2i = parameters['w2i']
i2w = parameters['i2w']
data = parameters['hyponyms']
gold = parameters['hypernyms']

print(len(data))

vocab_size = len(vocab)

# # Supervised Learning - Model Architecture

class HHD(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(HHD, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

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
        
        
        # a step from above
        # find similarity between query and negative samples
        batch_projT = torch.transpose(batch_proj,1,2) #bs*d*k
        simNegs = torch.bmm(neg_hypernymsT,batch_projT) #bs*ns*d x bs*d*k = bs*ns*k
        simNegsOutput = self.output(simNegs) #bs*ns*1
        simNegsOutput = torch.squeeze(simNegsOutput,2) #bs*ns
        
        
        
        # simPos - bs*1, simNegs - bs*ns
        return simPosOutput,simNegsOutput
    
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

# ### Projection model

projection_model = HHD(vocab_size,dim)
projection_model.to(device)
projection_model = torch.load("/kaggle/input/inlp-project/HH_Projection_model_1A.pt")
projection_model.eval()

'''
    predict function will take a query, a word and will return list of 
    100 closest words according to projection learning model ie supervised learning
'''
def predict_supervised(query):
    
    try:
        q = torch.tensor([w2i[query]]).to(device)
    except:
        return "word not found in vocab"
    
    closest_hypernyms = [] 
    
    h = torch.tensor(list(range(1,vocab_size))).to(device)
    s = projection_model.similarity(q,h,h.shape[0]) #bs*1

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

# # Unsupervised Learning

# ### Compute dictionary with key as hyponym and value as list of hypernyms

def compute_hyponym_hypernyms(data,gold):
  hyponym_classification = {}
  hyponym_hypernyms = {}
  for i in range(len(data)):
    hyponym = data[i]
    hypernyms = gold[i]
    hyponym_hypernyms[hyponym] = hypernyms
  return hyponym_hypernyms

hyponym_hypernyms_eng = compute_hyponym_hypernyms(data,gold)

# ### Compute dictionary with key as hypernym and value as list of hyponyms

def compute_hypernym_hyponyms(hyponym_hypernyms):

  hypernym_hyponyms = {}
  for i in hyponym_hypernyms:
    hypernyms_list = hyponym_hypernyms[i]
    for j in hypernyms_list:
      if j in hypernym_hyponyms:
        hypernym_hyponyms[j].append(i)
      else:
        hypernym_hyponyms[j] = [i]
  for i in hyponym_hypernyms:
    hypernyms_list = set(hyponym_hypernyms[i])
    hyponym_hypernyms[i] = hypernyms_list
  
  return hypernym_hyponyms

hypernym_hyponyms_eng = compute_hypernym_hyponyms(hyponym_hypernyms_eng)

# ### Load custom trained with negative sampling word2vec embeddings

# Word2Vec pretrained embeddings further trained with hypernym negative sampling

with open('/kaggle/input/inlp-project/hypernym-hyponym-embeddings_1A.pkl', 'rb') as f:
    emb = pickle.load(f)
    
word2vec_embeddings_eng = emb

queries_eng = data

len(queries_eng)

# ### Compute co-hyponyms for given set of hyponyms

Q = []
Hq = []
co_hyponyms_query = {}

# query -> hyponym
for query in queries_eng:                            
  # get hypernyms for a given hyponym 
  hypernyms_query = hyponym_hypernyms_eng[query]     
  co_hyponyms = []
  for hypernym in hypernyms_query:                   
    # store all hyponyms of the hypernyms derived from above in the list "co-hyponyms" 
    for hyponym in hypernym_hyponyms_eng[hypernym]:
        if hyponym != query:         
            # append co-hyponym only if it is not the original hyponym
            co_hyponyms.append(hyponym)
  
  #compute set of co-hyponyms list to get unique co-hyponyms
  co_hyponyms_set = set(co_hyponyms)        
  co_hyponyms_freq = {}
  # compute frequency of each co-hyponym of a given hyponym and store in co_hyponyms_query
  for co_hyponym in co_hyponyms_set:
    freq = co_hyponyms.count(co_hyponym)
    co_hyponyms_freq[co_hyponym] = freq

  co_hyponyms_query[query] = co_hyponyms_freq

# ### Compute cosine similarities

from numpy.linalg import norm

# cosine similarity with custom trained word2vec embeddings
def calculate_cosine_similarity_word2vec(a,b):
  A = np.zeros(300)
  B = np.zeros(300)
  if a in word2vec_embeddings_eng and b in word2vec_embeddings_eng:
    A = word2vec_embeddings_eng[a]
    B = word2vec_embeddings_eng[b]
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine
  else:
    return 0

# queries_eng

# ### Compute final set of hypernyms for given set of hyponyms

def compute_final_set_of_hypernyms(queries):
    
    final_set_of_hypernyms_given_query = {}

    for query in queries:
      # compute scores of each co-hyponym for given hyponyms
      # score is calculated using the formula: score = cosine_similarity(co-hyponym,hyponym) * frequency(co-hyponym)
      scores = {}
      for co_hyponym in co_hyponyms_query[query]:
        score = calculate_cosine_similarity_word2vec(query,co_hyponym)
        scores[co_hyponym] = score * co_hyponyms_query[query][co_hyponym]

      # append top(most similar) 15 co-hyponyms in Q
      Q = []
      Q.append(query)
      scores = sorted(scores.items(), key=lambda x:x[1], reverse = True)
      count = 0
      for i in scores:
#         if count == 15:
#           break
#         count += 1
        Q.append(i[0])
        
      # Hq contains the list of hypernyms of the top 15 co-hyponyms
      Hq = []
      for q in Q:
        Hq.extend(hyponym_hypernyms_eng[q])

      # Compute frequency of hypernym as the count of hyponyms for which it is a hypernym 
      hypernym_freq = {}
      for h in Hq:
        c = 0
        for cohyponym in co_hyponyms_query[query]:
          if h in hyponym_hypernyms_eng[cohyponym]:
            c += 1
        hypernym_freq[h] = c

      # Score each hypernym as follows: score = cosine_similarity(hypernym,original hyponym) * frequency(hypernym)^2
      hypernym_scores = {}
      Hq = set(Hq)
      for h in Hq:
        score = calculate_cosine_similarity_word2vec(query,h)
        hypernym_scores[h] = score * hypernym_freq[h] * hypernym_freq[h]


      # Take top 15 hypernyms as the final list of hypernyms for given set of hyponyms
      final_set_hypernyms = []
      hypernym_scores = sorted(hypernym_scores.items(), key=lambda x:x[1], reverse = True)
      count = 0
      for i in hypernym_scores:
#         if count == 15:
#           break
#         count += 1
        final_set_hypernyms.append(i[0])
      
      if len(final_set_hypernyms) > 100:
        final_set_hypernyms = final_set_hypernyms[:100]
        
      final_set_of_hypernyms_given_query[query] = final_set_hypernyms
      
    return final_set_of_hypernyms_given_query

def predict_unsupervised(word):
    if word not in data:
        return []
    else:
        ans = compute_final_set_of_hypernyms([word])[word]
        return ans

def top_15(hypernyms,input_query):
    ans = []
    scores = {}
    for h in hypernyms:
        sim = calculate_cosine_similarity_word2vec(h,input_query)
        scores[h] = sim

    scores = sorted(scores.items(), key=lambda x:x[1], reverse = True)
    count = 0
    for i in scores:
        if input_query != i[0]:
            if count == 15:
              break
            count += 1
            ans.append(i[0])
    return ans

def compute_hybrid_hypernyms(queries):
    top_15_hypernyms = []
    for q in tqdm(queries):
        hypernyms_supervised = predict_supervised(q)
        hypernyms_unsupervised = predict_unsupervised(q)
        hypernyms = []
        for h in hypernyms_supervised:
            hypernyms.append(h)
        hypernyms.extend(hypernyms_unsupervised)
        hypernyms = set(hypernyms)
        top15 = top_15(hypernyms,q)
        top_15_hypernyms.append(top15)
    return top_15_hypernyms

def write_to_file(final_list_hypernyms,filename):
    f = open(filename, "w")
    for i in final_list_hypernyms:
        hypernyms = ""
        for index,j in enumerate(i):
            if index == len(i)-1:
                hypernyms += str(j) + "\n"
            else:
                hypernyms += str(j) + "\t"
        f.write(hypernyms)
    f.close()

queries = data
final_hybrid_hypernyms = compute_hybrid_hypernyms(queries)
#final_hybrid_hypernyms[0]

# data[0]

def remove_underscores(final_hybrid_hypernyms):
    for ind_sent,h_sent in enumerate(final_hybrid_hypernyms):
        for ind_h,fh in enumerate(h_sent):
            if '_' in fh:
                h = fh.split('_')
                hyp = ""
                for i in range(len(h)):
                    if i == len(h)-1:
                        hyp += h[i]
                    else:
                        hyp += h[i] + " "
                h_sent[ind_h] = hyp
        final_hybrid_hypernyms[ind_sent] = h_sent
    return final_hybrid_hypernyms

final_hybrid_hypernyms = remove_underscores(final_hybrid_hypernyms)

write_to_file(final_hybrid_hypernyms,"hypernyms.1A.english.training.txt")

# ## Compute top 15 hypernyms given query hyponym

q = 'pollution'

print("Hyponym:",q)
print("Top 15 hypernyms from supervised model")
hypernyms_supervised = predict_supervised(q)
sup_top_15 = top_15(hypernyms_supervised,q)
print(sup_top_15)
print()
print("Top 15 hypernyms from unsupervised model")
hypernyms_unsupervised = predict_unsupervised(q)
unsup_top_15 = top_15(hypernyms_unsupervised,q)
print(unsup_top_15)
print()
print("Top 15 hypernyms from hybrid model")
hypernyms = []
for h in hypernyms_supervised:
    hypernyms.append(h)
hypernyms.extend(hypernyms_unsupervised)
hypernyms = set(hypernyms)
hypernyms_final = top_15(hypernyms,q)
print(hypernyms_final)

