# %% [code] {"id":"ASMaI2_SENSD","outputId":"c6ced464-a6cc-4621-feb2-67a5ca7e44b1","execution":{"iopub.status.busy":"2023-05-02T18:02:17.208013Z","iopub.execute_input":"2023-05-02T18:02:17.208422Z","iopub.status.idle":"2023-05-02T18:02:27.353607Z","shell.execute_reply.started":"2023-05-02T18:02:17.208390Z","shell.execute_reply":"2023-05-02T18:02:27.352574Z"}}
from math import log2
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import random
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report
import math

# %% [markdown]
# # Loading preprocessed dictionaries and lists

import pickle

with open('/kaggle/input/inlp-project/hypernym-hyponym-dictionaries_training.pkl', 'rb') as f:
    data = pickle.load(f)
    
    vocab_eng = data['vocab']
    i2w_eng = data['i2w']
    w2i_eng = data['w2i']
    data_eng = data['hyponyms']
    gold_eng = data['hypernyms']


# %% [markdown]
# # Compute dictionary with key as hyponym and value as list of hypernyms 

def compute_hyponym_hypernyms(data,gold):
  hyponym_classification = {}
  hyponym_hypernyms = {}
  for i in range(len(data)):
    hyponym = data[i]
    hypernyms = gold[i]
    hyponym_hypernyms[hyponym] = hypernyms
  return hyponym_hypernyms

hyponym_hypernyms_eng = compute_hyponym_hypernyms(data_eng,gold_eng)

# %% [markdown]
# # Compute dictionary with key as hypernym and value as list of hyponyms 

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
# hypernym_hyponyms_ital = compute_hypernym_hyponyms(hyponym_hypernyms_ital)
# hypernym_hyponyms_span = compute_hypernym_hyponyms(hyponym_hypernyms_span)

# %% [markdown]
# # Load pretrained Glove embeddings

# Glove Embeddings

embed_dict = {}

with open('/kaggle/input/glove-embeddings/glove.6B.300d.txt','r') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:],'float32')
    embed_dict[word]=vector

embed_dict['oov'] = np.zeros(300)
glove_embeddings_eng = embed_dict

# %% [markdown]
# # Load custom trained with negative sampling word2vec embeddings

# Word2Vec pretrained embeddings further trained with hypernym negative sampling

with open('/kaggle/input/inlp-project/hypernym-hyponym-embeddings_training.pkl', 'rb') as f:
    data = pickle.load(f)
    
word2vec_embeddings_eng = data

queries_eng = data_eng

# %% [markdown]
# # Compute co-hyponyms for given set of hyponyms

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

co_hyponyms_query['pollution']

# %% [markdown]
# # Compute cosine similarities

from numpy.linalg import norm

# cosine similarity with pretrained glove embeddings
def calculate_cosine_similarity_glove(a,b):
  A = np.zeros(300)
  B = np.zeros(300)
  if a in glove_embeddings_eng and b in glove_embeddings_eng:
    A = glove_embeddings_eng[a]
    B = glove_embeddings_eng[b]
    cosine = np.dot(A,B)/(norm(A)*norm(B))
#     print(cosine)
    return 1-cosine
  else:
    return 0

# cosine similarity with custom trained word2vec embeddings
def calculate_cosine_similarity_word2vec(a,b):
  A = np.zeros(300)
  B = np.zeros(300)
  if a in word2vec_embeddings_eng and b in word2vec_embeddings_eng:
    A = word2vec_embeddings_eng[a]
    B = word2vec_embeddings_eng[b]
    cosine = np.dot(A,B)/(norm(A)*norm(B))
#     print(cosine)
    return 1-cosine
  else:
    return 0

queries_eng_test = [data_eng[3]]
for q_test in queries_eng_test:
    print(q_test)
    print(gold_eng[3])

# %% [markdown]
# # Compute final set of hypernyms for given set of hyponyms

def compute_final_set_of_hypernyms(embedding,queries):
    
    final_set_of_hypernyms_given_query = {}

    for query in queries:
      # compute scores of each co-hyponym for given hyponyms
      # score is calculated using the formula: score = cosine_similarity(co-hyponym,hyponym) * frequency(co-hyponym)
      scores = {}
      for co_hyponym in co_hyponyms_query[query]:
        if embedding == "glove":
#             print("glove")
            score = calculate_cosine_similarity_glove(query,co_hyponym)
            scores[co_hyponym] = score * co_hyponyms_query[query][co_hyponym]
        else:
#             print("word2vec")
            score = calculate_cosine_similarity_word2vec(query,co_hyponym)
            scores[co_hyponym] = score * co_hyponyms_query[query][co_hyponym]
            
      # append top(most similar) 15 co-hyponyms in Q
      Q = []
      Q.append(query)
      scores = sorted(scores.items(), key=lambda x:x[1], reverse = True)
      count = 0
      for i in scores:
        if count == 15:
          break
        count += 1
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
#       print(Hq)
      for h in Hq:
        if embedding == "glove":
            score = calculate_cosine_similarity_glove(query,h)
        else:
            score = calculate_cosine_similarity_word2vec(query,h)
    
        hypernym_scores[h] = score * hypernym_freq[h] * hypernym_freq[h]


      # Take top 15 hypernyms as the final list of hypernyms for given set of hyponyms
      final_set_hypernyms = []
      hypernym_scores = sorted(hypernym_scores.items(), key=lambda x:x[1], reverse = True)
      count = 0
      for i in hypernym_scores:
        if count == 15:
          break
        count += 1
        final_set_hypernyms.append(i[0])
      
      final_set_of_hypernyms_given_query[query] = final_set_hypernyms
      
    return final_set_of_hypernyms_given_query,hypernym_scores

final_list_hypernyms_glove,sg = compute_final_set_of_hypernyms("glove",queries_eng_test)
final_list_hypernyms_word2vec,sw = compute_final_set_of_hypernyms("word2vec",queries_eng_test)

print("Hyponym:",data_eng[3])
print()
print("Given hypernyms:")
print(gold_eng[3])
print()
print("Hypernyms generated using Glove embeddings:")
print(final_list_hypernyms_glove[q_test])
print()
print("Hypernyms generated using custom trained Word2Vec embeddings:")
print(final_list_hypernyms_word2vec[q_test])

def predict(word):
    if word not in data_eng:
        return []
    else:
        ans = compute_final_set_of_hypernyms("word2vec",[word])
        return ans

input_query = 'pollution'
set_of_hypernyms,sim_scores = predict(input_query)
print(set_of_hypernyms[input_query])

def write_to_file(final_list_hypernyms,filename):
    f = open(filename, "w")
    for i in final_list_hypernyms:
      hyponym = i + " -> {"
      f.write(hyponym)
      hypernyms = ""
      count = len(final_list_hypernyms[i])
      c = 0
      for j in final_list_hypernyms[i]:
        if c != count-1:
          hypernyms += j + ", "
        else:
          hypernyms += j + "}" + "\n"
        c += 1
      f.write(hypernyms)
    f.close()

write_to_file(final_list_hypernyms_glove,"Hyponym_hypernyms_glove_eng.txt")
write_to_file(final_list_hypernyms_word2vec,"Hyponym_hypernyms_word2vec_eng.txt")