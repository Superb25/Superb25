#!/usr/bin/env python
# coding: utf-8

# In[ ]:


numbers = [2,4,6]
for val in numbers:
    val += 2
    print(val)


# In[2]:


import random


# In[6]:


## my coin_toss game
#1 = Head
#0 = Tail

coin_toss = random.randint(0,1)
if coin_toss == 1:
    print('Head')
elif coin_toss == 0:
        print('Tail')
else:
    print('invalid entry')


# In[ ]:




