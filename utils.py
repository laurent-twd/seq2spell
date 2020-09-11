import itertools
import tensorflow as tf
import tensorflow_probability as tfp

def flatten(l):
        return list(itertools.chain.from_iterable(l))

def text_processing(data, length_filter = 200):
    text = flatten(list(map(lambda s: s.lower().split('.'), data)))
    text = [t.strip() for t in text if len(t) > 10 and len(t) <= length_filter]
    tokenized_text = list(map(lambda x: x.split(), text))
    return tokenized_text, text

def create_noisy_text(text, char2idx):

    p_error = 0.1
    p_delete = 0.4
    p_insert = 0.05
    p_transpose = 0.4
    p_replace = 0.2

    p = [p_delete, p_insert, p_transpose, p_replace]
    dist_error = tfp.distributions.Bernoulli(probs = p_error)
    dist_type = tfp.distributions.Categorical(probs = p)
    n_char = len(char2idx)
    dist_characters  = tfp.distributions.Categorical(probs = tf.constant(1 / n_char, shape = (n_char)))

    characters = list(char2idx.keys())

    def create_noisy_sentence(sentence):

        i = 0
        n = len(sentence)
        new_sentence = sentence

        while i < n - 1 and i < len(new_sentence):
            is_error = dist_error.sample().numpy()
            if is_error:
                type_error = dist_type.sample()
                if type_error == 0: # delete
                    new_sentence = new_sentence[:i] + new_sentence[(i+1):]
                    i+=1
                elif type_error == 1: #insert
                    idx = dist_characters.sample()
                    new_sentence = new_sentence[:i] + characters[idx] + new_sentence[i:]
                    i += 2
                elif type_error == 2: # transpose
                    s_i = new_sentence[i]
                    s_i_1 = new_sentence[i+1]
                    new_sentence = new_sentence[:i] + s_i_1 + s_i + new_sentence[(i+2):]
                    i += 2
                else: # replace
                    idx = dist_characters.sample()
                    new_sentence = new_sentence[:i] + characters[idx] + new_sentence[(i+1):]
                    i += 1
            else:
                i+=1

        return new_sentence

    return list(map(create_noisy_sentence, text))
    


def prepare_dataset(chard2idx, noisy_text, text):

    def get_index_char(char):
        
        try:
            return char2idx[char]
        except:
            return 1

    source_text_indexes = list(map(lambda s: [1] + list(map(get_index_char, s)) + [2], noisy_text))
    target_text_indexes = list(map(lambda s: [1] + list(map(get_index_char, s)) + [2], text))

    length_source = list(map(len, source_text_indexes))
    length_target = list(map(len, source_text_indexes))

    paddded_source_text_indexes = tf.keras.preprocessing.sequence.pad_sequences(source_text_indexes, padding = 'post', maxlen = max(length_source))
    paddded_target_text_indexes = tf.keras.preprocessing.sequence.pad_sequences(target_text_indexes, padding = 'post', maxlen = max(length_target))

    return paddded_source_text_indexes, paddded_target_text_indexes, length_source, length_target