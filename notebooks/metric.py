#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# In[11]:


## top K

with open('../results/generate_v1_result/predict_results_None.txt') as f:
        lines = f.readlines()

        
# preds = []
probs = []
for l in lines[1:]:
    cur_result = l.strip().split('\t')
    if cur_result[0] == '0':
        # only store the prob of 1
        probs.append(1-float(cur_result[1]))
    else:
        probs.append(float(cur_result[1]))
sorted_probs_indices = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)

df = pd.read_csv('../data/novel_valid_generate_v1.csv')
labels = list(df['label'])
total = sum(labels)
print("total 1: ", total)

def get_top_k(labels, sorted_probs_indices, probs, k, total):
    correct = 0
    for i in range(k):
        index = sorted_probs_indices[i]
        if probs[index] < 0.5:
            print(probs[index]) # check probability
        if labels[index] == 1:
            correct += 1

    precision = correct / k
    recall = correct / total
    f1 = 2*precision*recall / (precision+recall)
    return precision, recall, f1

print(get_top_k(labels, sorted_probs_indices, probs, 5, total))
print(get_top_k(labels, sorted_probs_indices, probs, 10, total))
print(get_top_k(labels, sorted_probs_indices, probs, 15, total))
print(get_top_k(labels, sorted_probs_indices, probs, total, total))
        
    


# In[10]:


with open('../results/generate_v1_result/predict_results_None.txt') as f:
        lines = f.readlines()

preds = []
probs = []
for l in lines[1:]:
    cur_result = l.strip().split('\t')
    preds.append(int(cur_result[0]))
    probs.append(float(cur_result[1]))

df = pd.read_csv('../data/novel_valid_generate_v1.csv')
labels = list(df['label'])



pre = precision_score(labels, preds)
rec = recall_score(labels, preds)
f1 = f1_score(labels, preds)
conf_matrix = confusion_matrix(labels, preds)
print(pre, rec, f1)
conf_matrix


# In[6]:


# with open('result_v3/predict_results_None.txt') as f:
#         lines = f.readlines()

# preds = []
# probs = []
# for l in lines[1:]:
#     cur_result = l.strip().split('\t')
#     preds.append(int(cur_result[0]))
#     probs.append(float(cur_result[1]))

# df = pd.read_csv('result_v3/novel_valid_v3.csv')
# labels = list(df['label'])



# pre = precision_score(labels, preds)
# rec = recall_score(labels, preds)
# f1 = f1_score(labels, preds)
# conf_matrix = confusion_matrix(labels, preds)
# print(pre, rec, f1)
# conf_matrix


# In[21]:





# In[22]:





# In[30]:





# In[31]:




