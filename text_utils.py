import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import copy

"""
Implement a class object that should have the following functions:

1) object initialization:
This function should be able to take arguments of data directory, batch size and sequence length.
The initialization should be able to process data, load preprocessed data and create training and 
validation mini batches.

2)helper function to preprocess the text data:
This function should be able to do:
    a)read the txt input data using encoding='utf-8'
    b)
        b1)create self.char that is a tuple contains all unique character appeared in the txt input.
        b2)create self.vocab_size that is the number of unique character appeared in the txt input.
        b3)create self.vocab that is a dictionary that the key is every unique character and its value is a unique integer label.
    c)split training and validation data.
    d)save your self.char as pickle (pkl) file that you may use later.
    d)map all characters of training and validation data to their integer label and save as 'npy' files respectively.

3)helper function to load preprocessed data

4)helper functions to create training and validation mini batches

"""

def batch_generator(text, batch_size, seq_len):
    text = copy.copy(text)
    batch_area = batch_size * seq_len
    num_batches = int(len(text) / batch_area)
    text = text[:batch_area * num_batches]
    text = text.reshape((batch_size, -1))
    while True:
        np.random.shuffle(text)
        for l in range(0, text.shape[1], seq_len):
            x = text[:, l:l + seq_len]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

class TextLoader():
    def __init__(self, directory, batch_size, seq_len):
        self.directory = directory
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.preprocess(self.directory)
        
    def preprocess(self, directory):
        with open(os.path.join(directory, 'shakespeare.txt'), encoding='utf-8') as f:
            text = list(f.read().replace('\n', ''))
            char_set = set(text)
            self.vocab_size = len(char_set)
            self.vocab = dict(zip(list(char_set), range(self.vocab_size)))
            self.char = ['a'] * self.vocab_size
            for key, value in self.vocab.items():
                self.char[value] = key
            train_set, val_set = text[:int(0.85 * len(text))], text[int(0.85 * len(text)):]
            train = np.asarray([self.vocab[key] for key in train_set])
            val = np.asarray([self.vocab[key] for key in val_set])
            np.save('train_set.npy', train)
            np.save('val_set.npy', val)
            cPickle.dump(self.char, open('unique_char.pkl', 'wb'))
    
    def loadData(self):
        train_set = np.load('train_set.npy')
        val_set = np.load('val_set.npy')
        char = cPickle.load(open('unique_char.pkl', 'rb'))
        return train_set, val_set, char
    '''
    def getBatch(self, data, batch_i):
            batch_len = len(data) // self.batch_size
            num_batches = batch_len // self.seq_len
            X = np.zeros((self.batch_size, self.seq_len))
            y = np.zeros((self.batch_size, self.seq_len))
            for i in range(self.batch_size):
                X[i,:] = data[(i * batch_len + seq_len * batch_i):(i * batch_len + seq_len * (batch_i + 1))]
                if i * batch_len + seq_len * (batch_i + 1) + 1 < len(data):
                    y[i,:] = data[(i * batch_len + seq_len * batch_i + 1):(i * batch_len + seq_len * (batch_i + 1) + 1)]
                else:
                    y[i,:-1] = data[(i * batch_len + seq_len * batch_i + 1):len(data)]
                    y[-1] = 0
            return X, y
    '''
'''
directory = ''
batch_size = 32
seq_len = 50
load = TextLoader(directory, batch_size, seq_len)
train_set, val_set, char = load.loadData()
X, y = load.getBatch(train_set, 1)
'''