import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, n_char, vector_size, enc_units):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.range_cnn_kernel_size = range(2, 6)
    self.embedding = tf.keras.layers.Embedding(
                                            input_dim = n_char + 1, 
                                            output_dim = vector_size,
                                            mask_zero = True,
                                            trainable = True)
    
    self.gru_lr = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences = True,
                                   return_state = True,
                                   recurrent_initializer='glorot_uniform')
    
    self.gru_rl = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences = True,
                                   return_state = True,
                                   recurrent_initializer = 'glorot_uniform')    
    
    self.cnn = {}
    for kernel_size in self.range_cnn_kernel_size:
        self.cnn['cnn_kernel_{}'.format(kernel_size)] = tf.keras.layers.Conv1D(
                        filters = 2**kernel_size,
                        kernel_size = kernel_size, 
                        strides = 1, 
                        padding = 'valid',
                        activation = 'relu'
                        )

    self.projection = tf.keras.layers.Dense(self.enc_units, activation = 'linear')

  def call(self, padded_char_lr, hidden_states):
    
    padded_char_rl = tf.reverse(padded_char_lr, axis = [1])
    x_lr = self.embedding(padded_char_rl)
    x_rl = self.embedding(padded_char_rl)

    # Bi-GRU network
    mask_lr = self.embedding.compute_mask(padded_char_lr)
    mask_rl = self.embedding.compute_mask(padded_char_rl)

    x_lr, h_lr = self.gru_lr(x_lr, initial_state = hidden_states, mask = mask_lr)
    x_rl, h_rl = self.gru_rl(x_lr, initial_state = hidden_states, mask = mask_rl)

    x = tf.concat([x_lr, tf.reverse(x_rl, axis = [1])], axis = 2)

    # CNN feature vectors

    cnn_features = []
    for k in self.range_cnn_kernel_size:
        features = self.cnn['cnn_kernel_{}'.format(k)](x_lr)
        new_mask = tf.cast(tf.expand_dims(mask_lr[:, (k - 1):], axis = 2), dtype = tf.float32)
        features = features - new_mask * 1e9
        features = tf.reduce_max(features, axis = 1)
        cnn_features.append(features)
    cnn_features = tf.concat(cnn_features, axis = 1)

    gru_cnn_features = tf.concat([h_lr, h_rl, cnn_features], axis = 1)

    return x, self.projection(gru_cnn_features)

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

    score += (tf.expand_dims(mask, axis = 2) * -1e9) 
    attention_weights = tf.nn.softmax(score, axis = 1) 

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis = 1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, vector_size, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size + 3, 
                                               output_dim = vector_size, 
                                               mask_zero = False,
                                               trainable = True)

    self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    
    self.dense_h = tf.keras.layers.Dense(dec_units)
    self.dense_context = tf.keras.layers.Dense(dec_units, activation = 'linear', name = 'context_concat')
    self.fc = tf.keras.layers.Dense(vocab_size + 3, name = 'dense_output')
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, inputs, initial_states, enc_outputs, mask):

    x = self.embedding(inputs)
    x, h, c = self.lstm(x, initial_state = initial_states)
    context_vector, attention_weights = self.attention(h, enc_outputs, mask)

    x = tf.reshape(x, (-1, x.shape[2]))
    x = tf.concat([x, context_vector], axis = 1)
    x = self.fc(x)

    return x, [h, c], attention_weights

