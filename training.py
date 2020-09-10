
import pandas as pd
from utils import * #flatten, text_processing, prepare_dataset
from model import *
from collections import Counter
import tensorflow as tf 
import os

################
#     DATA     #
################

path_data = r'/Users/laurentthanwerdas/Documents/Documents/Etudes/NY/Personal/PROJECTS/Deep_Embedded_Clustering/severeinjury.csv'
data = list(pd.read_csv(path_data, encoding = 'latin9')['text'])
tokenized_text, text = text_processing(data)
lengths = list(map(len, text))

frequency = Counter(flatten(tokenized_text))
vocabulary = set(list(frequency.keys()))
vocab_size = len(vocabulary)

characters = set(list(Counter(flatten(text)).keys()))
n_char = len(characters)

word2idx = dict(zip(vocabulary, range(4, vocab_size + 4)))
char2idx = dict(zip(characters, range(2, n_char + 2)))

paddded_text_indexes, padded_tokenized_text_indexes = prepare_dataset(word2idx, char2idx, tokenized_text, text)

ds_X = tf.data.Dataset.from_tensor_slices(paddded_text_indexes)
ds_Y = tf.data.Dataset.from_tensor_slices(padded_tokenized_text_indexes)
dataset = tf.data.Dataset.zip((ds_X, ds_Y))

####################
#     TRAINING     #
####################

batch_size = 16
epochs = 1

ds = dataset.batch(batch_size)
ds = ds.shuffle(len(paddded_text_indexes))

for _ in range(epochs):
    progbar = tf.keras.utils.progbar(len(paddded_text_indexes))
    for padded_char_lr, padded_target_words, in ds:
        batch_loss = train_step(padded_char_lr, padded_target_words, True)
        values = [('Loss', batch_loss)]
        progbar.add(padded_char_lr.shape[0])










