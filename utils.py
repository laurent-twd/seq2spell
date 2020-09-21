import itertools
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def flatten(l):
        return list(itertools.chain.from_iterable(l))

def text_processing(data, length_filter = 200):
    text = flatten(list(map(lambda s: s.lower().split('.'), data)))
    text = [t.strip() for t in text if len(t) > 10 and len(t) <= length_filter]
    return text

keyboard = [list("qwertyuiop"), list("asdfghjkl"), list("zxcvbnm,"), list('~~     ~~')]
keyboardCoordinates = {}
for i in range(len(keyboard)):
    for j in range(len(keyboard[i])):
        try:
            keyboardCoordinates[keyboard[i][j]].append(np.array((i, j)))
        except:
            keyboardCoordinates[keyboard[i][j]] = [np.array((i, j))]

keyboardCoordinates.pop('~')

def get_distance(key1, key2):
    def dist(c1, c2):
        dK = np.abs(c1 - c2).sum()
        if dK == 0 or dK > 2:
            return 10
        else:
            return dK
    try:
        l1 = keyboardCoordinates[key1] ; l2 = keyboardCoordinates[key2]
        d = list(map(lambda y: list(map(lambda x: dist(x, y), l1)), l2))
        return np.min(d)
    except:
        return 10

def get_noisy_text(text, char2idx, p_error = 0.04, p_delete = 0.3, p_insert = 0.1, p_transpose = 0.3, p_replace = 0.3):

    p = [p_delete, p_insert, p_transpose, p_replace]

    characters = [c for c in list(char2idx.keys()) if c.isalpha()] + [' ']
    set_characters = set(characters)
    c2i = dict(zip(characters, range(len(characters))))
    range_char = range(len(characters))

    p_distance_characters = list(map(lambda y: list(map(lambda x: get_distance(x, y), characters)), characters))
    p_distance_characters = tf.constant(p_distance_characters, dtype = tf.float32)
    p_distance_characters = tf.math.softmax(- p_distance_characters)
        
    def quick_noisy_sentence(sentence):

        i = 0
        n = len(sentence)
        new_sentence = ''

        is_error = np.random.binomial(1, p_error, size = n)
        type_errors = np.random.choice(range(4), p = p, size = n)
        j = 0
        while i < n - 1:
            if is_error[i] and sentence[i] in set_characters:
                type_error = type_errors[j]
                j+=1
                if type_error == 0: # delete
                    i+=1
                elif type_error == 1: #insert
                    idx = c2i[sentence[i]]
                    idx = np.random.choice(range_char, p = p_distance_characters[idx].numpy())
                    new_sentence += characters[idx] + sentence[i]
                    i += 1
                elif type_error == 2: # transpose
                    s_i = sentence[i]
                    s_i_1 = sentence[i+1]
                    new_sentence += s_i_1 + s_i
                    i += 2
                else: # replace
                    idx = c2i[sentence[i]]
                    idx = np.random.choice(range_char, p = p_distance_characters[idx].numpy())
                    new_sentence += characters[idx]
                    i += 1
            else:
                new_sentence += sentence[i]
                i+=1

        return new_sentence
    
    return list(map(quick_noisy_sentence, text))
    
def prepare_dataset(chard2idx, noisy_text, text):

    def get_index_char(char):
        
        try:
            return char2idx[char]
        except:
            return 3

    source_text_indexes = list(map(lambda s: [1] + list(map(get_index_char, s)) + [2], noisy_text))
    target_text_indexes = list(map(lambda s: [1] + list(map(get_index_char, s)) + [2], text))

    length_source = list(map(len, source_text_indexes))
    length_target = list(map(len, source_text_indexes))

    paddded_source_text_indexes = tf.keras.preprocessing.sequence.pad_sequences(source_text_indexes, padding = 'post', maxlen = max(length_source))
    paddded_target_text_indexes = tf.keras.preprocessing.sequence.pad_sequences(target_text_indexes, padding = 'post', maxlen = max(length_target))

    return paddded_source_text_indexes, paddded_target_text_indexes, length_source, length_target

