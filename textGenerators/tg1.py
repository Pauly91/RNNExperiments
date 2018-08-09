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