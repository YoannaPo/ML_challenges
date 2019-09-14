#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import Word2Vec

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# # Dane 

# In[2]:


sentences = []

for i in range(100000):
    start = np.random.randint(0, 10)
    finish = start + np.random.randint(3, 20)
    sentence = [str(x) for x in range(start, finish)]
    
    sentences.append(sentence)


# # Model Word2Vec

# In[3]:


model = Word2Vec(sentences, size=10, window=5, min_count=1)


# In[4]:


model.wv['1']


# In[5]:


def plot_heatmap(model):
    plt.figure(figsize=(15,8))
    sns.heatmap( model.wv[model.wv.vocab], linewidths=0.5 );


# In[7]:


plot_heatmap(Word2Vec(sentences, size=50, window=5))


# # PCA
# 

# In[19]:


X.shape


# In[29]:


def plot_pca(model):
    X = model.wv[model.wv.vocab]
    pca_model = PCA(n_components=2)
    result = pca_model.fit_transform(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(result[:,0], result[:,1]);

    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))


# In[30]:


plot_pca(Word2Vec(sentences, size=50, window=5))


# # Podobne slowa

# In[32]:


model.wv.most_similar('10', topn=3)


# In[33]:


model.wv.most_similar(positive=['10', '8'], negative=['1'], topn=3)


# In[35]:


10 + 8 - 1


# In[17]:


Y = model.wv[model.wv.vocab]
pca_model = PCA(n_components=1)
pca_model.fit_transform(Y)


# In[ ]:




