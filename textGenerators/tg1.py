import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

# Tutorial is based on: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
# Data is from : https://raw.githubusercontent.com/pranjal52/text_generators/master/sonnets.txt

'''
This is a character level mapping based text generetor

Read this after:

It must also be noted here that I have used character level mappings and not word 
mappings. However, when compared with each other, a word-based model shows much 
higher accuracy as compared to a character-based model. This is because the latter 
model requires a much larger network to learn long-term dependencies as it not only 
has to remember the sequences of words, but also has to learn to predict a 
grammatically correct word. However, in case of a word-based model, the latter has 
already been taken care of.

But since this is a small dataset (with 17,670 words), and the number of unique words 
(4,605 in number) constitute around one-fourth of the data, it would not be a wise 
decision to train on such a mapping. This is because if we assume that all unique
words occurred equally in number (which is not true), we would have a word occurring 
roughly four times in the entire training dataset, which is just not sufficient to 
build a text generator.

 

'''

text=(open("data/text.txt").read())
text=text.lower()

characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

X = []
 Y = []
length = len(text)
seq_length = 100
  for i in range(0, length-seq_length, 1):
     sequence = text[i:i + seq_length]
     label =text[i + seq_length]
     X.append([char_to_n[char] for char in sequence])
     Y.append(char_to_n[label])


X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)
model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_modified, Y_modified, epochs=100, batch_size=50)
model.save_weights('text_generator_700_0.2_700_0.2_100.h5')
model.load_weights('text_generator_700_0.2_700_0.2_100.h5')

string_mapped = X[99]
full_string = [n_to_char[value] for value in string_mapped]

# generating characters
for i in range(seq_length):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))
    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

txt=""
for char in full_string:
    txt = txt+char
txt