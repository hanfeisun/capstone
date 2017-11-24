import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
import json
import os
# import seaborn as sns
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import h5py
from pandas_ml import ConfusionMatrix
import langdetect

from keras.layers import Input, Embedding, LSTM, Dense, BatchNormalization
from keras.models import Model
import keras

# %matplotlib inline

np.random.seed(0)
BASE_DIR = ''
GLOVE_DIR = 'glove.6B.100d.txt'
MAX_SEQUENCE_LENGTH = 40
MAX_SENTENCE = 10
MAX_NB_WORDS = 20000
EMBEDDING_DIM = int(''.join([s for s in GLOVE_DIR.split('/')[-1].split('.')[-2] if s.isdigit()]))  # 100
VALIDATION_SPLIT = 0.1
PRELOAD = False


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


def safe_detect(s):
    try:
        global langdetect_count
        langdetect_count += 1
        if langdetect_count % 10000 == 0:
            print("Detected languages for  {} reviews".format(count))
        return langdetect.detect(s)
    except:
        return 'unknown'


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
    sentence_seq = sentence.split("\n")
    L =  len(sentence_seq)
    if L >= n:
        return sentence_seq[:n]
    else:
        return [""] * (n - L) + sentence_seq

pad_sentence = df_rev_balanced.text.map(lambda x:truncate_or_pad(x, MAX_SENTENCE)).values

flatten_pad_sentence = np.concatenate(pad_sentence).ravel()

seqs = tokenizer.texts_to_sequences(flatten_pad_sentence)
seqs = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)
seqs = np.array(seqs).reshape([-1,MAX_SENTENCE, MAX_SEQUENCE_LENGTH])




X = seqs
Y = df_rev_balanced.rating.values.astype(int)
Y_cat = to_categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=VALIDATION_SPLIT, random_state=9)

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

filepath = "imp-balanced-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('training_history.csv')
history = History()
callbacks_list = [checkpoint, history, csv_logger]

in_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sentence = Embedding(input_dim=nb_words,
                              output_dim=EMBEDDING_DIM,
                              input_length=MAX_SEQUENCE_LENGTH,
                              weights=[embedding_matrix],
                              trainable=False)(in_sentence)

gru_sentence = GRU(50, return_sequences=True)(embedded_sentence)
gru_sentence = Dropout(0.2)(gru_sentence)
gru_sentence = GRU(50)(gru_sentence)
gru_sentence = Dropout(0.2)(gru_sentence)
encoded_model = Model(in_sentence, gru_sentence)
print(encoded_model.summary())

sequence_input = Input(shape=(MAX_SENTENCE, MAX_SEQUENCE_LENGTH), dtype='int32', name='sentences')
seq_encoded = TimeDistributed(encoded_model)(sequence_input)
seq_encoded = Dropout(0.2)(seq_encoded)
seq_encoded = GRU(50)(seq_encoded)

x = seq_encoded
x = BatchNormalization()(x)
x = Dense(20, activation='relu')(x)

x = BatchNormalization()(x)
x = Dense(20, activation='relu')(x)

gru_output = Dense(6, activation='softmax', name='softmax_output')(x)
gru_output = Dense(1, activation='sigmoid', name='scalar_output')(gru_output)

model = Model(inputs=[sequence_input], outputs=[gru_output])
print(model.summary())

import tensorflow as tf

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
#               metrics=[keras.metrics.categorical_accuracy, "accuracy"])

model.compile(loss=keras.losses.binary_crossentropy, optimizer='rmsprop',
              metrics=[keras.metrics.mse, keras.metrics.binary_crossentropy])

y_train = np.argmax(y_train, axis=1).reshape([-1, 1])
y_test = np.argmax(y_test, axis=1).reshape([-1, 1])
y_train = y_train / 5.
y_test = y_test / 5.


model.fit({'sentences': X_train},
          {'scalar_output': y_train},
          epochs=30,
          batch_size=128,
          validation_data=({'sentences': X_test}, {'scalar_output': y_test}),
          callbacks=callbacks_list
          )
#
y_test_predict = model.predict({'sentences': X_test})

import scipy

correlation = scipy.stats.pearsonr(y_test.ravel(), y_test_predict.ravel())
print(correlation)