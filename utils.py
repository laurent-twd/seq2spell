import itertools
import tensorflow as tf

def flatten(l):
        return list(itertools.chain.from_iterable(l))

def text_processing(data, length_filter = 200):
    text = flatten(list(map(lambda s: s.lower().split('.'), data)))
    text = [t.strip() for t in text if len(t) > 10 and len(t) <= length_filter]
    tokenized_text = list(map(lambda x: x.split(), text))
    return tokenized_text, text

def prepare_dataset(word2idx, chard2idx, tokenized_text, text):

    def get_index_word(word):
        
        try:
            return word2idx[word]
        except:
            return 3

    def get_index_char(char):
        
        try:
            return char2idx[char]
        except:
            return 1

    text_indexes = list(map(lambda s: list(map(get_index_char, s)), text))
    tokenized_text_indexes = list(map(lambda s: [1] + list(map(get_index_word, s)) + [2], tokenized_text))

    maxlen = max(list(map(len, text_indexes)))
    paddded_text_indexes = tf.keras.preprocessing.sequence.pad_sequences(text_indexes, padding = 'post', maxlen = maxlen)
    maxlen = max(list(map(len, tokenized_text_indexes)))
    padded_tokenized_text_indexes = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text_indexes, padding = 'post', maxlen = maxlen)

    return paddded_text_indexes, padded_tokenized_text_indexes