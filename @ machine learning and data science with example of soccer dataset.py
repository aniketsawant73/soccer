#!/usr/bin/env python
# coding: utf-8

# #steps for machine learning(what a ml specialist do?)
# 1 import the data
# 2 clean the data
# 3 split data into training set(80%) and test set(20%)  (data parsing)
# 4 create a model(model creation)
# 5 check the output of model using test set.(accuracy testing)
# 6 improve the model

# # 1 import the data

# In[1]:


import pandas as pd
data_frame = pd.read_csv('soccer.csv')


# In[2]:


data_frame.shape


# In[3]:


data_frame.describe()


# In[4]:


data_frame.values


# In[5]:


#data_frame[data_frame["Age"]>40].head()


# # which players are very good with less salary?

# In[6]:


df1 = pd.DataFrame(data_frame, columns=['Name','Wage','Value'])
df1


# In[ ]:





# In[7]:


df1 = pd.DataFrame(data_frame, columns=['Name','Wage','Value'])
def value_to_float(x):
    if type(x) == float or type(x) == int:
        return x
    if 'K' in x:
        if len(x) > 1:
            return float(x.replace('K', '')) * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0
    if 'B' in x:
        return float(x.replace('B', '')) * 1000000000
    return 0.0
wage = df1['Wage'].replace('[\€,]','', regex=True).apply(value_to_float)
value = df1['Value'].replace('[\€,]','', regex=True).apply(value_to_float)

df1['Wage']= wage
df1['Value']= value

df1['difference'] = df1['Value'] - df1['Wage']
df1


# In[8]:


df1.sort_values('difference', ascending=False)


# In[22]:


import seaborn as sns
sns.set()

graph = sns.scatterplot(x='Wage', y='Value', data=df1)
graph


# In[29]:


from bokeh.plotting import figure,show
from bokeh.models import HoverTool

TOOLTIPS= HoverTool(tooltips=[
    ("index","$index"),
     ("(Wage,Value)","(@Wage,@Value)"),
     ("Name","@Name")]
                   )

p = figure(title='Soccer 2019', x_axis_label='Wage', y_axis_label='Value', plot_width=700, plot_height=700, tools=[TOOLTIPS])
p.circle('Wage', 'Value', size=10, source=df1)
show(p)


# In[ ]:





# In[ ]:




