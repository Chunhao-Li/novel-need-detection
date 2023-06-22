#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import pandas as pd


# In[2]:


# df = pd.read_csv('data/novel_test_modified.csv')
# df.head()


# In[3]:


df_neg = pd.read_excel('../data/annotation_sentence.xlsx')


# In[4]:


neg_sentences = []
for _,row in df_neg.iterrows():
    for col in df_neg.columns:
        if isinstance(row[col], str):
            neg_sentences.append(row[col])
neg_sentences = list(set(neg_sentences))
print(len(neg_sentences))
neg_labels = [0]*len(neg_sentences)
neg_sentences[:3]


# In[5]:


df_new = pd.read_excel('../data/new_novel_need.xlsx')
df_new


# In[6]:


pos_sentences = []

for _,row in df_new.iterrows():
#     pos_sentences.append(row['text'])
#     pos_sentences.append(row['prompt1'])
#     pos_sentences.append(row['prompt2'])
    pos_sentences.append(row['generate_v1'])
print(len(pos_sentences))
pos_labels = [1]*len(pos_sentences)


# In[7]:


# df = pd.DataFrame.from_dict({'sentence1':pos_sentences+neg_sentences, 'label': pos_labels+neg_labels})
# df = df.sample(n=len(df))

# df['label'].value_counts()
# df_train, df_valid = train_test_split(df, train_size = 0.8, random_state=1)
# print(df_valid.label.value_counts())
# print(df_train.label.value_counts())
# df_train.to_csv('data/novel_train_v2.csv', index=False)
# df_valid.to_csv('data/novel_valid_v2.csv', index=False)


# In[8]:


df_unlabel = pd.read_excel('../data/reviews 2020-23.xlsx', sheet_name=0)
unlabel_sentences = []
for _,row in df_unlabel[3:].iterrows():
    for col in df_unlabel.columns[1:]:
        if isinstance(row[col], str):
            processed = row[col][row[col].find('):')+2:].strip()
            if len(processed) == 0:
                continue
            unlabel_sentences.append(processed)
unlabel_sentences = list(set(unlabel_sentences))
unlabel_labels = [0]*len(unlabel_sentences)
print(len(unlabel_sentences))


df_unlabel2 = pd.read_excel('../data/reviews 2020-23.xlsx', sheet_name=1)
unlabel_sentences2 = []
for _,row in df_unlabel2[3:].iterrows():
    for col in df_unlabel2.columns[1:]:
        if isinstance(row[col], str):
            processed = row[col][row[col].find('):')+2:].strip()
            if len(processed) == 0:
                continue
            unlabel_sentences2.append(processed)
unlabel_sentences2 = list(set(unlabel_sentences2))
unlabel_labels2 = [0]*len(unlabel_sentences2)
print(len(unlabel_sentences2))


# In[10]:


df = pd.DataFrame.from_dict({'sentence1':pos_sentences+neg_sentences+unlabel_sentences+unlabel_sentences2
                                , 'label': pos_labels+neg_labels+unlabel_labels+unlabel_labels2})
df = df.sample(n=len(df))

df_train, df_valid = train_test_split(df, train_size = 0.8, random_state=1)
print(df_valid.label.value_counts())
print(df_train.label.value_counts())
# df_train.to_csv('../data/novel_train_prompt1.csv', index=False)
# df_valid.to_csv('../data/novel_valid_prompt1.csv', index=False)

# df_train.to_csv('../data/novel_train_prompt2.csv', index=False)
# df_valid.to_csv('../data/novel_valid_prompt2.csv', index=False)

df_train.to_csv('../data/novel_train_generate_v1.csv', index=False)
df_valid.to_csv('../data/novel_valid_generate_v1.csv', index=False)


# In[26]:


# # df = df.astype({"sentence1": str, "label_unused": int, "label": int})
# df['sentence1'] = df['sentence1'].astype('str')


# In[58]:




