import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
import json
import os
import collections
import scipy.sparse as sp

import itertools
import numpy as np
# import seaborn
import operator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import joblib
from sklearn.model_selection import train_test_split

from keras.callbacks import History
from keras.datasets import imdb
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM, GRU
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger

from keras.layers import Input, Embedding, LSTM, Dense, BatchNormalization
from keras.models import Model
import keras

from keras import layers

# %matplotlib inline

np.random.seed(0)
BASE_DIR = ''
GLOVE_DIR = 'glove.6B.100d.txt'
MAX_WORD_PER_SENTENCE = 128
MAX_SENTENCE_PER_SESSION = 32
MAX_NB_WORDS = 4096
EMBEDDING_DIM = int(''.join([s for s in GLOVE_DIR.split('/')[-1].split('.')[-2] if s.isdigit()]))  # 100
VALIDATION_SPLIT = 0.1
PRELOAD = False

REGRESSION = False

SENTENCE_EMBEDDING_SIZE = None
SESSION_EMBEDDING_SIZE = None


def load_glove_into_dict(glove_path):
    """
    :param glove_path: strpath
    loads glove file into a handy python-dict representation, where a word is a key with a corresponding N-dim vector
    http://nlp.stanford.edu/data/glove.6B.zip (pretrained-embeddings)
    """
    embeddings_ix = {}
    with open(glove_path) as glove_file:
        for line in glove_file:
            val = line.split()
            word = val[0]
            vec = np.asarray(val[1:], dtype='float32')
            embeddings_ix[word] = vec
    return embeddings_ix


def get_features_for_layer(X, trained_model, layer_number, batches=256):
    """
    :param X: Batch with dimensions according to the models first layer input-shape
    :param trained_model: Model to extract data from
    :param layer_number: Index of the layer we want to extract features from.
    :param batches: If set it will call the function in batches to save (gpu)memory
    :return:
    """

    get_features = K.function([trained_model.layers[0].input, K.learning_phase()],
                              [trained_model.layers[layer_number].output])

    features = get_features([X, 0])

    return features


def array_batch_yield(X, group_size):
    for i in range(0, len(X), group_size):
        yield X[i:i + group_size]


def minority_balance_dataframe_by_multiple_categorical_variables(df, categorical_columns=None, downsample_by=0.1):
    """
    :param df: pandas.DataFrame
    :param categorical_columns: iterable of categorical columns names contained in {df}
    :return: balanced pandas.DataFrame
    """
    if categorical_columns is None or not all([c in df.columns for c in categorical_columns]):
        raise ValueError('Please provide one or more columns containing categorical variables')

    minority_class_combination_count = df.groupby(categorical_columns).apply(lambda x: x.shape[0]).min()

    minority_class_combination_count = int(minority_class_combination_count * downsample_by)

    df = df.groupby(categorical_columns).apply(
        lambda x: x.sample(minority_class_combination_count)
    ).drop(categorical_columns, axis=1).reset_index().set_index('level_1')

    df.sort_index(inplace=True)

    return df


def split_train_test_set(df):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    return train, test


langdetect_count = 0

df_reviews = pd.read_csv('oneperline.csv')  # , encoding='utf-8')
df_reviews['len'] = df_reviews.text.str.len()
df_reviews['rating'] = df_reviews['rating'].round()

df_reviews = df_reviews[df_reviews['len'].between(10, 4000)]
#     df_reviews = df_reviews[df_reviews.language == 'en']
# balancing dataset
df_rev_balanced = minority_balance_dataframe_by_multiple_categorical_variables(
    df_reviews,
    categorical_columns=['rating'],
    downsample_by=1
)

df_rev_balanced.to_csv('balanced_reviews.csv', encoding='utf-8')

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df_rev_balanced.text.tolist())
joblib.dump(tokenizer, 'tokenizer.pickle')

WORD_INDEX_SORTED = sorted(tokenizer.word_index.items(), key=operator.itemgetter(1))


def truncate_or_pad(sentence, n):
    sentence_seq = list(map(lambda x: x.strip(), sentence.split("\n")))
    L = len(sentence_seq)
    if L >= n:
        return sentence_seq[:n]
    else:
        return [""] * (n - L) + sentence_seq


pad_sentence = df_rev_balanced.text.map(lambda x: truncate_or_pad(x, MAX_SENTENCE_PER_SESSION)).values


def sentence_embedding_func(last_str, str):
    global SENTENCE_EMBEDDING_SIZE
    words = len(str.split(" "))

    if str.startswith("AAAAA"):
        embed_speaker = [0, 1]
    elif str.startswith("UUUUU"):
        embed_speaker = [1, 0]
    else:
        embed_speaker = [0, 0]

    if str.startswith("AAAAA") or str.startswith("UUUUU"):
        str = str[8:]

    embed_sentence_length = [min(len(str), 400), ]
    embed_word_length = [min(len(str.strip().split(" ")), 100), ]

    pos_word = ["love", "friend"]
    neg_word = ["stupid", "idiot", "fuck"]
    pos = False
    neg = False
    for p in pos_word:
        if p in str:
            pos = True
            break

    for n in neg_word:
        if n in str:
            neg = True
            break

    embed_sentiment = [1 if pos else 0, 1 if neg else 0]

    if len(last_str) == 0 or len(str) == 0:
        embed_overlap = [0,
                         0,
                         0,
                         0,
                         0,
                         0]
        embed_uniq_rate = [0, 0]
    else:
        this_vocab = set(str.strip("\n").split(" "))
        last_vocab = set(str.strip("\n").split(" "))
        embed_overlap = [1,
                         len(this_vocab),
                         len(last_vocab),
                         len(this_vocab | last_vocab),
                         len(this_vocab & last_vocab),
                         len(this_vocab | last_vocab) / len(this_vocab & last_vocab),
                         ]

        embed_uniq_rate = [1, len(this_vocab) / words]

    if "wh" in str or "how" in str:
        embed_question = [1]
    else:
        embed_question = [0]

    ret = embed_speaker + embed_sentence_length + embed_word_length + embed_sentiment + embed_uniq_rate + embed_question + embed_overlap
    SENTENCE_EMBEDDING_SIZE = len(ret)
    return ret


def session_embedding_func(str):
    global SESSION_EMBEDDING_SIZE

    words = len(str.replace("\n", " ").split(" "))
    turns = len(str.split("\n"))
    embed_session_length = [min(len(str), 100), min(len(str), 1000), min(len(str), 5000)]
    embed_session_words = [min(words, 100), min(words, 1000), min(words, 5000)]
    embed_session_turns = [turns]
    pos_word = ["love", "friend"]
    neg_word = ["stupid", "idiot", "fuck"]
    pos = False
    neg = False
    for p in pos_word:
        if p in str:
            pos = True
            break

    for n in neg_word:
        if n in str:
            neg = True
            break

    embed_sentiment = [1 if pos else 0, 1 if neg else 0]

    embed_uniq_rate = [len(set(str.replace("\n", " ").split(" "))) / words]

    ret = embed_session_length + embed_session_words + embed_session_turns + embed_sentiment + embed_uniq_rate
    SESSION_EMBEDDING_SIZE = len(ret)
    return ret


X_sentence_aux_embedding = np.array(
    [[sentence_embedding_func(last_sentence, sentence) for last_sentence, sentence in
      zip([""] + session[:-1:], session[::])] for session in pad_sentence], dtype="float32")
X_sentence_aux_embedding /= np.max(X_sentence_aux_embedding, axis=(0, 1,))

X_session_aux_embedding = np.array([session_embedding_func(session) for session in df_rev_balanced.text.values],
                                   dtype="float32")
X_session_aux_embedding /= np.max(X_session_aux_embedding, axis=(0,))

flatten_pad_sentence = np.concatenate(pad_sentence).ravel()

seqs = tokenizer.texts_to_sequences(flatten_pad_sentence)
seqs = pad_sequences(seqs, maxlen=MAX_WORD_PER_SENTENCE)
seqs = np.array(seqs).reshape([-1, MAX_SENTENCE_PER_SESSION, MAX_WORD_PER_SENTENCE])

X = seqs
Y = df_rev_balanced.rating.values.astype(int)
Y_cat = to_categorical(Y)
X_train_glove, X_test_glove, y_train, y_test = train_test_split(X, Y_cat, test_size=VALIDATION_SPLIT, random_state=9)
X_train_seten, X_test_seten, _, _ = train_test_split(X_sentence_aux_embedding, Y_cat,
                                                     test_size=VALIDATION_SPLIT,
                                                     random_state=9)
X_train_sessi, X_test_sessi, _, _ = train_test_split(X_session_aux_embedding, Y_cat,
                                                     test_size=VALIDATION_SPLIT,
                                                     random_state=9)

X_train = {
    "glove": X_train_glove,
    "sentence": X_train_seten,
    "session": X_train_sessi
}

X_test = {
    "glove": X_test_glove,
    "sentence": X_test_seten,
    "session": X_test_sessi,
}

# X_train_aux = np.array([i[1] for i in X_train], dtype='float32')
# X_train_aux = X_train_aux / np.max(X_train_aux, axis=0)
# X_train = np.array([i[0] for i in X_train])

# X_test_aux = np.array([i[1] for i in X_test], dtype='float32')
# X_test_aux = X_test_aux / np.max(X_test_aux, axis=0)
# X_test = np.array([i[0] for i in X_test])

# prepare embedding matrix
embedding_index = load_glove_into_dict(GLOVE_DIR)
nb_words = min(MAX_NB_WORDS, len(WORD_INDEX_SORTED))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

df_rev_balanced.groupby('rating')['len'].hist(alpha=0.1)
print('number of words in GLOVE: {}'.format(len(embedding_index)))
print(WORD_INDEX_SORTED[0:100:10])

if REGRESSION:
    filepath = "imp-balanced-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

else:
    filepath = "imp-balanced-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# csv_logger = CSVLogger('training_history.csv')
# history = History()
# callbacks_list = [checkpoint, history, csv_logger]
callbacks_list = [checkpoint]
import tensorflow as tf
from keras import initializers
from keras.engine.topology import Layer, InputSpec


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.glorot_normal
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_normal",
                                 trainable=True)

        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)

        weights = ai / tf.expand_dims(K.sum(ai, axis=1), 1)
        weighted_input = x * weights

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


in_sentence = Input(shape=(MAX_WORD_PER_SENTENCE,), dtype='int32')
sentence = layers.Lambda(lambda x: x[:, :MAX_SENTENCE_PER_SESSION])(in_sentence)
e = Embedding(input_dim=nb_words,
              output_dim=EMBEDDING_DIM,
              input_length=MAX_WORD_PER_SENTENCE,
              weights=[embedding_matrix],
              trainable=False)(sentence)

gru_output = e
if False:
    gru_output = GRU(50, return_sequences=True)(gru_output)
gru_output = layers.Bidirectional(GRU(50, return_sequences=True))(gru_output)
gru_output = TimeDistributed(Dense(100))(gru_output)
gru_output = AttLayer()(gru_output)

# gru_output = layers.concatenate([GRU(50)(gru_output), tf.keras.backend.constant([[1]])])
encoded_model = Model(inputs=[in_sentence], outputs=[gru_output])
print(encoded_model.summary())

sequence_input = Input(shape=(MAX_SENTENCE_PER_SESSION, MAX_WORD_PER_SENTENCE), dtype='int32', name='glove')
sentence_embedding_input = Input(shape=(MAX_SENTENCE_PER_SESSION, SENTENCE_EMBEDDING_SIZE), dtype='float32',
                                 name='sentence')
session_embedding_input = Input(shape=(SESSION_EMBEDDING_SIZE,), dtype='float32', name='session')

naive = True
if not naive:
    seq_encoded = TimeDistributed(encoded_model)(sequence_input)
    seq_encoded = layers.concatenate([seq_encoded, sentence_embedding_input], axis=2)
else:
    seq_encoded = sentence_embedding_input
# seq_encoded = Dropout(0.2)(seq_encoded)
if False:
    seq_encoded = layers.Bidirectional(GRU(32, return_sequences=True))(seq_encoded)
    seq_encoded = AttLayer()(seq_encoded)
else:
    seq_encoded = Dense(100)(layers.Flatten(seq_encoded))
seq_encoded = layers.concatenate([seq_encoded, session_embedding_input], axis=1)
x = seq_encoded
gru_output = Dense(32, activation='relu')(x)
gru_output = BatchNormalization()(gru_output)
gru_output = Dense(16, activation='relu')(gru_output)
gru_output = BatchNormalization()(gru_output)
gru_output = Dense(6, activation='softmax', name='softmax_output')(gru_output)
if REGRESSION:
    gru_output = Dense(1, activation='sigmoid', name='scalar_output')(gru_output)

model = Model(inputs=[sequence_input, sentence_embedding_input, session_embedding_input], outputs=[gru_output])
print(model.summary())

if REGRESSION:
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.rmsprop(),
                  metrics=[keras.metrics.mse, keras.metrics.binary_crossentropy])
else:
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.rmsprop(),
                  metrics=[keras.metrics.categorical_crossentropy, "accuracy"])

if REGRESSION:
    y_train = np.argmax(y_train, axis=1).reshape([-1, 1])
    y_test = np.argmax(y_test, axis=1).reshape([-1, 1])
    y_train = y_train / 5.
    y_test = y_test / 5.

    model.fit(X_train,
              {'scalar_output': y_train},
              epochs=60,
              batch_size=128,
              validation_data=(X_test, {'scalar_output': y_test}),
              callbacks=callbacks_list
              )
else:
    model.fit(X_train,
              {'softmax_output': y_train},
              epochs=30,
              batch_size=128,
              validation_data=(X_test, {'softmax_output': y_test}),
              callbacks=callbacks_list
              )
# #
# y_test_predict = model.predict({'sentences': X_test})
#
# import scipy
#
# correlation = scipy.stats.pearsonr(y_test.ravel(), y_test_predict.ravel())
# print(correlation)
