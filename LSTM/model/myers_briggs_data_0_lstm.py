import pandas as pd
import numpy as np
import re
import tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import matplotlib.pyplot as plt

df = pd.read_csv('../datasets/0.csv')
df_text = np.array(df['text'])
df_label = np.array(df['label'])

df_user = []
personality_types = ['intj', 'intp', 'entp', 'entj', 'infj', 'infp', 'enfj', 'enfp', 'istj', 'isfj', 'estj', 'esfj', 'istp', 'isfp', 'estp', 'esfp']

for i in df_text:
    text = i.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'https\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+','', text)
    for j in personality_types:
        text = re.sub(j, '', text)
    while True:
        before_text = text
        text=text.replace('  ', ' ')
        if before_text == text:
            break
    df_user.append(np.array(text.split('|||')))

df_user = np.array(df_user)

df_user_flattened = []
df_label_flattened = []

for index, i in enumerate(df_user):
    for j in i:
        df_user_flattened.append(j)
        df_label_flattened.append(df_label[index])

df_user_flattened = np.array(df_user_flattened)

lens = []
for i in df_user_flattened:
    sample_lens = i.split(" ")
    lens.append(len(sample_lens))

seed = 23
np.random.seed = seed
ds_df = pd.DataFrame()
ds_df['posts']=df_user_flattened
ds_df['labels'] = df_label_flattened
ds = ds_df.to_numpy()
np.random.shuffle(ds)
front_cutoff = int(ds.shape[0]*0.6)
mid_cutoff = int(ds.shape[0]*0.8)
ds_train, ds_val, ds_test = ds[:front_cutoff], ds[front_cutoff:mid_cutoff], ds[mid_cutoff:]

ds_train_texts, ds_train_labels = ds_train[:,0], ds_train[:,1]

ds_val_texts, ds_val_labels = ds_val[:, 0], ds_val[:, 1]

ds_test_texts, ds_test_labels = ds_test[:, 0], ds_test[:, 1]

vocab_size = 4000 

oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size
                      ,oov_token= oov_tok
                      )
tokenizer.fit_on_texts(ds_train_texts)
word_index = tokenizer.word_index

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# load in tokenizer
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

word_index = tokenizer.word_index

tokenized_train = tokenizer.texts_to_sequences(ds_train_texts)

tokenized_val = tokenizer.texts_to_sequences(ds_val_texts)

tokenized_test = tokenizer.texts_to_sequences(ds_test_texts)

padded = pad_sequences(tokenized_train, maxlen=150, padding='post', truncating='post')

padded_val = pad_sequences(tokenized_val, maxlen=150, padding='post', truncating='post')

padded_test = pad_sequences(tokenized_test, maxlen=150, padding='post', truncating='post')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(4000, 16, input_length=150),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.0001
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['accuracy', 'mse', 'mae', 'mape'])
model.summary()

model.fit(x=padded, y=np.asarray(ds_train_labels).astype('float32'), batch_size=8, epochs=5, validation_data=(padded_val, np.asarray(ds_val_labels).astype('float32')))

predictions = model.predict(padded_test)

labels_test = np.asarray(ds_test_labels).astype('float32')

model.evaluate(predictions, labels_test)

count = 0.0
correct = 0.0
for i in range(len(predictions)):
    predict = 0
    if predictions[i] < 0.5:
        predict = 0
    else:
        predict = 1
    if predict == labels_test[i]:
        correct += 1.0
    
    count += 1.0

print('test accuracy: ' + str(correct/count*100) +'%')

num_pos = 0
index_pos = []
index_correct_pos = []
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        num_pos += 1
        index_pos.append(i)
        if labels_test[i] == 1.0:
            index_correct_pos.append(i)
            
print('number of positive predictions: ' + str(num_pos))
print('\nindices of positive predictions:')
print(index_pos)
print('\nindices of correct positive predictions:')
print(index_correct_pos)

model.save('saved_model')

