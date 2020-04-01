
# coding: utf-8

# In[1]:


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random

import json
with open('intents.json') as file:
    data = json.load(file)


# In[2]:



words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])


# In[3]:


words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))


# In[4]:


labels = sorted(labels)


# In[5]:


training = []
output = []


# In[6]:


out_empty = [0 for _ in range(len(labels))]


# In[7]:


for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


# In[8]:


training = numpy.array(training)
output = numpy.array(output)


# In[9]:


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


# In[10]:


model.load('./model.tflearn')


# In[13]:


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        start = 0
        end =0
        for i in range(len(responses[0])):
            if(responses[0][i]=='['):
                start = i
            if(responses[0][i]==']'):
                end = i
        if(start==0):
            print(random.choice(responses))
        else:
            responses_ =""
            for i in range(start+1,end):
                responses_+= responses[0][i] 
       
            lst = []
            lst = responses_.split(",")
            print("Here are the courses for",inp)
            for i in range(len(lst)):
                print(lst[i])
        

chat()

