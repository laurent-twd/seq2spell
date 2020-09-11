
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

vector_size = 128
enc_units = 256
dec_units = enc_units

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
encoder = Encoder(100, vector_size, enc_units)
decoder = Decoder(50000, vector_size, dec_units)
optimizer = tf.keras.optimizers.Adam(1e-4)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype = loss_.dtype)
    loss_ *= mask
    return loss_

def train_step(padded_char_lr, padded_target_words, training):

    with tf.GradientTape() as tape:
        hidden_states = tf.constant(0., shape = (padded_char_lr.shape[0], encoder.enc_units))
        x, h =  encoder(padded_char_lr, hidden_states)
        dec_hidden = h, tf.constant(0., shape = (padded_char_lr.shape[0], decoder.dec_units))
        mask_input = tf.cast(tf.not_equal(padded_char_lr, 0), dtype = tf.float32)
        mask_output = tf.cast(tf.not_equal(padded_target_words, 0), dtype = tf.float32)

        batch_loss = []
        for t in range(padded_target_words.shape[1] - 1):
            dec_input = tf.expand_dims(padded_target_words[:, t], 1)
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, x, mask_input)
            loss = tf.expand_dims(loss_function(padded_target_words[:, t + 1], predictions), axis = 1)
            batch_loss.append(loss)
        batch_loss = tf.concat(batch_loss, axis = 1)
        batch_loss = tf.reduce_sum(batch_loss * mask_output[:, 1:], axis = 1) / (tf.reduce_sum(mask_output, axis = 1) - 1.)
        batch_loss = tf.reduce_mean(batch_loss)

        variables = encoder.trainable_variables + decoder.trainable_variables 
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

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










