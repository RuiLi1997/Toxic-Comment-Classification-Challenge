# -*- coding: utf-8 -*-

"""
Include rnn model building and training
tokenizer is saved in saved_token/rnn_tokenizer.pickle
model is saved in saved_model/my_rnn_model
training log is saved in logs/fit_rnn using tensorflow

"""


import tensorflow as tf
import pandas as pd
import numpy as np
import data_helper
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import datetime
import pickle

train_set = pd.read_csv("dataset/cleaned_train.csv")

PADDING_LENGTH = 100
embed_size = 50
FEATURE_PATH = 'features/glove.6B.50d.txt'

train_text = train_set.comment_text

# train label processing
train_labels = train_set.values[:,2:]
train_labels = np.asarray(train_labels, dtype=int)

x_train_clean, x_test_clean, y_train, y_test = train_test_split(train_text, train_labels, test_size=.2, shuffle=True)

x_train_clean = x_train_clean.tolist()
x_test_clean = x_test_clean.tolist()

# Tokenize the comment_text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text.tolist())

x_train_seq = tokenizer.texts_to_sequences(x_train_clean)
x_train_pad = pad_sequences(x_train_seq, maxlen=PADDING_LENGTH)

x_test_seq = tokenizer.texts_to_sequences(x_test_clean)
x_test_pad = pad_sequences(x_test_seq, maxlen=PADDING_LENGTH)

word_index = tokenizer.word_index

max_features = len(word_index)+1

embedding_matrix = data_helper.create_embedding_matrix(word_index, FEATURE_PATH)

# saving toeknizer

with open('saved_token/rnn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

tf.keras.backend.clear_session()


inp = Input(shape=(100, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.AUC(), 
                       tf.keras.metrics.Recall(), 
                       tf.keras.metrics.Precision()])

log_dir = "logs/fit_rnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(x_train_pad, y_train, batch_size=128, epochs=3, validation_data=(x_test_pad, y_test), shuffle=True, callbacks=[tensorboard_callback])
# Save the entire model as a SavedModel.
model.save('saved_model/my_rnn_model')

