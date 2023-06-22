#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed, BertConfig, BertTokenizer, BertForSequenceClassification,
)
import torch
import numpy as np


# In[13]:


device = torch.device('cuda')
model_path = '/home/lichunhao/models/bert-base-uncased/'
config = AutoConfig.from_pretrained(model_path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    from_tf=False,
    config=config,
)


model.to(device)


# In[14]:


batch_sentences = ["Hello I'm a single sentence",
                    "And another sentence",
                    "And the very very last one"]

tokenizer(batch_sentences)


# In[15]:


# from transformers import pipeline

# feature_extraction = pipeline('feature-extraction', model="/home/lichunhao/novel_need/prompt1_result/", tokenizer="/home/lichunhao/novel_need/prompt1_result/")
# features = feature_extraction(["Hello I'm a single sentence",
#                                "And another sentence",
#                                "And the very very last one"])


# In[16]:


import torch
import pandas as pd


# In[17]:


df_train = pd.read_csv('/home/lichunhao/novel_need/data/novel_train_generate_v1.csv')
df_valid = pd.read_csv('/home/lichunhao/novel_need/data/novel_valid_generate_v1.csv')


# In[18]:


pos_sentences = []
neg_sentences = []
pos_sentences2 = []
neg_sentences2 = []
for _,row in df_valid.iterrows():
    if row['label']==0:
        neg_sentences.append(row['sentence1'])
    else:
        pos_sentences.append(row['sentence1'])
        
        
for _,row in df_train.iterrows():
    if not isinstance(row['sentence1'], str):
        continue
    if row['label']==0:
        neg_sentences2.append(row['sentence1'])
    else:
        pos_sentences2.append(row['sentence1'])


# In[19]:


from tqdm import tqdm
pos_embeddings=[]
neg_embeddings=[]

for sent in tqdm(pos_sentences+pos_sentences2):
    input_ids = torch.tensor(tokenizer.encode(sent, padding='max_length', max_length=128, truncation=True)).unsqueeze(0)
    outputs = model(input_ids.to(device))
    pos_embeddings.append(outputs[0].detach().cpu().numpy())
    
for sent in tqdm(neg_sentences+neg_sentences2):
    input_ids = torch.tensor(tokenizer.encode(sent, padding='max_length', max_length=128, truncation=True)).unsqueeze(0)
    outputs = model(input_ids.to(device))
    neg_embeddings.append(outputs[0].detach().cpu().numpy())
    


# In[20]:


pos_embeddings = np.array(pos_embeddings).reshape(len(pos_embeddings),2)
neg_embeddings = np.array(neg_embeddings).reshape(len(neg_embeddings),2)


# In[10]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# tsne = TSNE(n_components=2, learning_rate='auto',
#                   init='random', perplexity=3)

# pos_embedded = tsne.fit_transform(pos_embeddings)
# neg_embedded = tsne.fit_transform(neg_embeddings)


# In[11]:


# fig = plt.figure()
# ax = plt.subplot(111)
# plt.scatter(pos_embedded[:,0], pos_embedded[:, 1], c='blue')
# plt.scatter(neg_embedded[:,0], neg_embedded[:, 1], c='red')
# # plt.scatter(pos_embeddings[:,0], pos_embeddings[:, 1], c='blue')
# # plt.scatter(neg_embeddings[:,0], neg_embeddings[:, 1], c='red')


# In[21]:


## generate_v1 vanilla model
fig = plt.figure()
ax = plt.subplot(111)
# plt.scatter(pos_embedded[:,0], pos_embedded[:, 1], c='blue')
# plt.scatter(neg_embedded[:,0], neg_embedded[:, 1], c='red')
plt.scatter(pos_embeddings[:,0], pos_embeddings[:, 1], c='blue')
plt.scatter(neg_embeddings[:,0], neg_embeddings[:, 1], c='red')


# In[12]:


## generate_v1 finetuned model
fig = plt.figure()
ax = plt.subplot(111)
# plt.scatter(pos_embedded[:,0], pos_embedded[:, 1], c='blue')
# plt.scatter(neg_embedded[:,0], neg_embedded[:, 1], c='red')
plt.scatter(pos_embeddings[:,0], pos_embeddings[:, 1], c='blue')
plt.scatter(neg_embeddings[:,0], neg_embeddings[:, 1], c='red')


# In[21]:


fig = plt.figure()
ax = plt.subplot(111)
# plt.scatter(pos_embedded[:,0], pos_embedded[:, 1], c='blue')
# plt.scatter(neg_embedded[:,0], neg_embedded[:, 1], c='red')
plt.scatter(pos_embeddings[:,0], pos_embeddings[:, 1], c='blue')
plt.scatter(neg_embeddings[:,0], neg_embeddings[:, 1], c='red')


# In[74]:


fig = plt.figure()
ax = plt.subplot(111)
# plt.scatter(pos_embedded[:,0], pos_embedded[:, 1], c='blue')
# plt.scatter(neg_embedded[:,0], neg_embedded[:, 1], c='red')
plt.scatter(pos_embeddings[:,0], pos_embeddings[:, 1], c='blue')
plt.scatter(neg_embeddings[:,0], neg_embeddings[:, 1], c='red')


# In[76]:


from transformers import pipeline

feature_extraction = pipeline('feature-extraction', model="v3_result", tokenizer="v3_result")
pos_features = feature_extraction(pos_sentences)

# neg_features = feature_extraction(neg_sentences)


# In[86]:


# [emb[0] for emb in pos_features]


# In[91]:


np.array([np.array(x) for x in pos_features[2][0]]).shape

