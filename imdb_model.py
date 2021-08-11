import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb, reuters
from tensorflow.keras.layers import Dense, Embedding, Input, CuDNNLSTM
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

INDEX_FROM = 3
NUM_WORDS = 10000
max_len = 100


def print_sentence(tokenized_sentence: str, id2w: dict):
    print(' '.join(id2w[_] for _ in tokenized_sentence))
    print('')
    print(tokenized_sentence)


def mapping_word_id(data):
    w2id = data.get_word_index()
    w2id = {k: (v + INDEX_FROM) for k, v in w2id.items()}
    w2id["<PAD>"] = 0
    w2id["<START>"] = 1
    w2id["<UNK>"] = 2
    w2id["<UNUSED>"] = 3
    id2w = {v: k for k, v in w2id.items()}
    return w2id, id2w


def get_dataset(dataset: str = 'imdb', max_len: int = 100):
    if dataset == 'imdb':
        data = imdb
    elif dataset == 'reuters':
        data = reuters
    else:
        raise NotImplementedError

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    w2id, id2w = mapping_word_id(data)

    (X_train, y_train), (X_test, y_test) = data.load_data(
        num_words=NUM_WORDS, index_from=INDEX_FROM)
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    # restore np.load for future normal usage
    np.load = np_load_old

    return (X_train, y_train), (X_test, y_test), (w2id, id2w)


def imdb_model(X: np.ndarray, num_words: int = NUM_WORDS, emb_dim: int = 128,
               lstm_dim: int = 128, output_dim: int = 2) -> tf.keras.Model:
    inputs = Input(shape=(X.shape[1:]), dtype=tf.float32)
    x = Embedding(num_words, emb_dim)(inputs)
    x = CuDNNLSTM(lstm_dim)(x)
    outputs = Dense(output_dim, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def train_model(X_train, y_train, X_test, y_test):
    model = imdb_model(X=X_train, num_words=NUM_WORDS, emb_dim=256, lstm_dim=128, output_dim=2)
    model.fit(X_train, y_train, batch_size=32, epochs=2,
              shuffle=True, validation_data=(X_test, y_test))
    return model

def grad_x_input_imdb(model, input_):
    m = model_embedding_input(model, 2)
    batch_size = min(len(input_), 5000)
    all_vals = np.empty((0,100,256))
    input_ = input_.reshape((-1,100,256))
    for i in range(int(len(input_) / batch_size)):
        inputs_ = tf.convert_to_tensor(input_[i*batch_size:(i+1)*batch_size], dtype='float32')
        with tf.GradientTape() as t:
            t.watch(inputs_)
            output = m(inputs_)
            gradients = t.gradient(output, inputs_, output_gradients=tf.one_hot(tf.argmax(output, axis=1), depth=2))
            all_vals = np.concatenate((all_vals, gradients.numpy() * inputs_), axis=0)
        print(len(all_vals))

    return all_vals.reshape((-1,25600))

def get_embedding(model, input_, labels=None):
    input_ = K.constant(input_)
    Embedding = tf.keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
    emb = tf.keras.layers.Flatten()(Embedding(input_))
    return emb

def watch_layer(layer, tape):
    """
    Make an intermediate hidden `layer` watchable by the `tape`.
    After calling this function, you can obtain the gradient with
    respect to the output of the `layer` by calling:

        grads = tape.gradient(..., layer.result)

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Store the result of `layer.call` internally.
            layer.result = func(*args, **kwargs)
            # From this point onwards, watch this tensor.
            tape.watch(layer.result)
            # Return the result to continue with the forward pass.
            return layer.result
        return wrapper
    layer.call = decorator(layer.call)
    return layer

def model_embedding_input(model, embed_layer_idx):
    input_shape = model.layers[embed_layer_idx].get_input_shape_at(0)  # get the input shape of desired layer
    print(input_shape)
    layer_input = Input(shape=input_shape[1:])  # a new input tensor to be able to feed the desired layer

    # create the new nodes for each layer in the path
    x = layer_input
    for layer in model.layers[embed_layer_idx:]:
        x = layer(x)

    # create the model
    new_model = tf.keras.Model(layer_input, x)
    return new_model


def grad_x_input(model, input_):
    batch_size = 1000
    all_vals = np.empty((0,28,28,1))
    input_ = input_.reshape((-1,28,28,1))

    for i in range(int(len(input_) / batch_size)):
        inputs_ = tf.convert_to_tensor(input_[i*batch_size:(i+1)*batch_size])
        model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)
        with tf.GradientTape() as t:
            t.watch(inputs_)
            output = model(inputs_)
            gradients = t.gradient(output, inputs_, output_gradients=tf.one_hot(tf.argmax(output, axis=1), depth=10))
            all_vals = np.concatenate((all_vals, gradients.numpy() * inputs_), axis=0)
        print(len(all_vals))

    return all_vals.reshape((-1,28*28))



def grad_x_input_autoencoder(model, input_):
    batch_size = 1000
    all_vals = np.empty((0,28,28,1))
    input_ = input_.reshape((-1,28,28,1))

    for i in range(int(len(input_) / batch_size)):
        inputs_ = tf.convert_to_tensor(input_[i*batch_size:(i+1)*batch_size])
        model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)
        with tf.GradientTape() as t:
            t.watch(inputs_)
            output = model(inputs_)
            gradients = t.gradient(output, inputs_, output_gradients=tf.ones_like(output))
            all_vals = np.concatenate((all_vals, gradients.numpy()), axis=0)
        print(len(all_vals))

    return all_vals.reshape((-1,28*28))

def run_training():
    (X_train, y_train), (X_test, y_test), (word2token, token2word) = \
        get_dataset(dataset='imdb', max_len=max_len)
    model = train_model(X_train, y_train, X_test, y_test)
    model.save('imdb_lstm')

#run_training()