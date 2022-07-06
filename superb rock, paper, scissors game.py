#!/usr/bin/env python
# coding: utf-8

# In[14]:


import random


# In[17]:


user_inputs = input("Enter('rock', 'paper', 'scissors'): ").lower()
Expected_outcome = ['rock', 'paper', 'scissors']
computer_inputs = random.choice(Expected_outcome)
print(f'user choses {user_inputs} and the computer has chosen {computer_inputs}')
      
if user_inputs == computer_inputs:
      print('its a draw')
elif user_inputs == "rock":
    if computer_inputs == "scissors":
        print('you win!')
    else:
        print('you lose!')
        
elif user_inputs == "paper":
    if computer_inputs == "rock":
        print('you win!')
    else:
        print('you lose!')
               
elif user_inputs == "scissors":
    if computer_inputs == "paper":
        print('you win!')
    else:
        print('you lose!')

