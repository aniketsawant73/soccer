#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
iris = load_iris()


# assuming sklearn provided clean data
# here we are doing supervised learning

# In[2]:


X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names
target_names


# In[3]:


type(X)


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[5]:


print(X_train.shape)
print(X_test.shape)


# In[6]:


from sklearn.tree import DecisionTreeClassifier
knn = DecisionTreeClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[7]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[8]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[9]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[13]:


sample = [[3,5,4,2], [2,3,5,4]]
predictions = knn.predict(sample)
pred_species= [iris.target_names[p] for p in predictions]
print("predictions:", pred_species)


# In[ ]:




