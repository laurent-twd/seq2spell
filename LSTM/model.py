import tensorflow as tf



class Encoder(tf.keras.Model):
  def __init__(self, n_char, vector_size, enc_units):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(input_dim = n_char + 4, 
                                               output_dim = vector_size,
                                               mask_zero = True,
                                               trainable = True)
    
    self.lstm_1 = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences = True,
                                   return_state = True,
                                   recurrent_initializer='glorot_uniform',
                                   dropout = 0.1,
                                   recurrent_dropout = 0.1)
    
    self.lstm_2 = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences = True,
                                   return_state = True,
                                   recurrent_initializer = 'glorot_uniform',
                                   dropout = 0.1,
                                   recurrent_dropout = 0.1)
    
  def call(self, padded_char_lr, hidden_states, training):
    
    x = self.embedding(padded_char_lr)
    mask_lr = self.embedding.compute_mask(padded_char_lr)
    x, h1, c1 = self.lstm_1(x, initial_state = hidden_states, mask = mask_lr, training = training)
    x, h2, c2 = self.lstm_2(x, initial_state = hidden_states, mask = mask_lr, training = training)

    return x, [[h1, c1], [h2, c2]]

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values, mask):

    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    new_mask = 1. - mask
    score += (tf.expand_dims(new_mask, axis = 2) * -1e9) 
    attention_weights = tf.nn.softmax(score, axis = 1) 

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis = 1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, n_char, vector_size, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(input_dim = n_char + 4, 
                                               output_dim = vector_size, 
                                               mask_zero = False,
                                               trainable = True)

    self.lstm_1 = tf.keras.layers.LSTM(self.dec_units,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform',
                                      dropout = 0.1,
                                      recurrent_dropout = 0.1)

    self.lstm_2 = tf.keras.layers.LSTM(self.dec_units,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform',
                                      dropout = 0.1,
                                      recurrent_dropout = 0.1)
    
    self.fc = tf.keras.layers.Dense(n_char + 4, name = 'dense_output')
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, inputs, initial_states, enc_outputs, mask, training):

    x = self.embedding(inputs)
    x, h1, c1 = self.lstm_1(x, initial_state = initial_states[0], training = training)
    x, h2, c2 = self.lstm_2(x, initial_state = initial_states[1], training = training)

    context_vector, attention_weights = self.attention(h2, enc_outputs, mask)

    x = tf.reshape(x, (-1, x.shape[2]))
    x = tf.concat([x, context_vector], axis = 1)
    x = self.fc(x)

    return x, [[h1, c1], [h2, c2]], attention_weights


