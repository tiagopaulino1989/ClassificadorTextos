#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

texto = ['They are novels','have you read this book', 'who is the author','what are the characters',
    'This is how I bought the book','I like fictions','what is your favorite book','This is my book']
classe = ['stmt','question','question','question','stmt','stmt','question','stmt']

df1=pd.DataFrame(texto, columns=['Texto'])
df2=pd.DataFrame(classe, columns=['Classe'])
data=pd.concat([df1,df2],axis=1)

# Convert categorical variable to numeric (0 for question, 1for stmt)
data["Class_cleaned"]=np.where(data["Classe"]=="question",0,1)

#Model based in Naive Bayes Multinomial
classifier = MultinomialNB()
#Vectorizing de Text from messages for pre-processing
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['Texto'])
targets = data['Class_cleaned'].values 

#fiting Model
classifier.fit(counts,targets)


#Variable to Predict
x_pred = ['what do you mean']
x_pred_counts = vectorizer.transform(x_pred)

predict = classifier.predict(x_pred_counts)

print('-----------------------------------')
if predict[0]==0:
    print('The text class is: question')
elif predict[0]==0:
    print('The text class is: stmt')
print('-----------------------------------')

