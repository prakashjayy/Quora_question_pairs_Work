import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Merge, Input, concatenate, dot, Flatten, Reshape, Bidirectional, add
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text

import argparse
import pickle

parser = argparse.ArgumentParser(description='Matching two sentences')
parser.add_argument('-idl', '--input_data_loc', help='', default='data/model_test.csv')
parser.add_argument('-mw', '--model_weights', default="models/base_model5")
args = parser.parse_args()

max_len = 25
word_index = 78990
DROPOUT=0.1

print("[Load the data]")
test = pd.read_csv(args.input_data_loc)
print("[There are a total of {} data points to evaluate]".format(test.shape[0]))

test["question1"] = test["question1"].apply(lambda x: str(x))
test["question2"] = test["question2"].apply(lambda x: str(x))
test["question1"] = test["question1"].apply(lambda x: x.replace("'", ""))
test["question2"] = test["question2"].apply(lambda x: x.replace("'", ""))

print("[Load the tokenizer]")
with open("data/tokenizer_embedding.pkl", "rb") as files:
    tk_train, embedding_matrix, _ = pickle.load(files)

print("[Pre-Processing the data]")
x1_test = tk_train.texts_to_sequences(test.question1.values)
x1_test = sequence.pad_sequences(x1_test, maxlen=max_len)

x2_test = tk_train.texts_to_sequences(test.question2.values.astype(str))
x2_test = sequence.pad_sequences(x2_test, maxlen=max_len)


print("[Building the network]")
question1 = Input(shape=(max_len,))
question2 = Input(shape=(max_len,))

q1 = Embedding(word_index + 1,
                 300,
                 weights=[embedding_matrix],
                 input_length=max_len,
                 trainable=False)(question1)
q1 = Bidirectional(LSTM(128, return_sequences=True), merge_mode="sum")(q1)

q2 = Embedding(word_index + 1,
                 300,
                 weights=[embedding_matrix],
                 input_length=max_len,
                 trainable=False)(question2)
q2 = Bidirectional(LSTM(128, return_sequences=True), merge_mode="sum")(q2)

attention = dot([q1,q2], [1,1])
attention = Flatten()(attention)
attention = Dense((max_len*128))(attention)
attention = Reshape((max_len, 128))(attention)

merged = add([q1,attention])
merged = Flatten()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)

is_duplicate = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[question1,question2], outputs=is_duplicate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print("[Load the weights]")
model.load_weights(args.model_weights)

print("[Start Evaluating the test data... Takes time Please be patience...................................]")
loss, accuracy = model.evaluate([x1_test, x2_test], test.is_duplicate.values, verbose=0)
print("Test Accuracy: ", accuracy)
