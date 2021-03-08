#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2


# In[2]:


os.chdir('E:\\New folder (4)')


# # Function to read the images

# In[7]:


def get_image(path):
    image=cv2.imread(path)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(90,135),cv2.INTER_LINEAR)
    img=gray[:100,:]
    return img


# In[8]:


images=[]
labels=[]


# # Reading the images from the folders

# In[9]:


for i in ['1','2','3','4']:
    for j in enumerate(os.listdir(os.getcwd()+'\\'+i)):
        imag=get_image(os.getcwd()+'\\'+i+'\\'+j[1])
        images.append(imag)
        labels.append(i)
        


# In[11]:


import numpy as np


# In[12]:


images=np.array(images)


# In[13]:


images.shape


# In[15]:


pixels=images.flatten().reshape(380,9000)


# In[17]:


pixels.shape


# # Using the Kmean Clustering

# In[20]:


from sklearn.cluster import KMeans


# In[21]:


km=KMeans(n_clusters=4,init='k-means++')


# In[22]:


km.fit(pixels)


# In[23]:


results=pd.DataFrame({'Clusters_labels':km.labels_})


# In[24]:


results


# In[26]:


clust_3=results[results.Clusters_labels==3]


# # Seeing the images that belong to cluster 3

# In[28]:


for i in clust_3.index:
    plt.imshow(images[i,:],cmap='gray')
    plt.show()


# In[29]:


import sklearn.metrics as mt


# # Final silhouette_score

# In[30]:


mt.silhouette_score(pixels,km.labels_)


# In[ ]:




