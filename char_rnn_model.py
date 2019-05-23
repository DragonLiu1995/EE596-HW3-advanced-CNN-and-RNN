import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import os
"""
TO: Define your char rnn model here

You will define two functions inside the class object:

1) __init__(self, args_1, args_2, ... ,args_n):

    The initialization function receives all hyperparameters as arguments.

    Some necessary arguments will be: batch size, sequence_length, vocabulary size (number of unique characters), rnn size,
    number of layers, whether use dropout, learning rate, use embedding or one hot encoding,
    and whether in training or testing,etc.

    You will also define the tensorflow operations here. (placeholder, rnn model, loss function, training operation, etc.)


2) sample(self, sess, char, vocab, n, starting_string):
    
    Once you finish training, you will use this function to generate new text

    args:
        sess: tensorflow session
        char: a tuple that contains all unique characters appeared in the text data
        vocab: the dictionary that contains the pair of unique character and its assoicated integer label.
        n: a integer that indicates how many characters you want to generate
        starting string: a string that is the initial part of your new text. ex: 'The '

    return:
        a string that contains the genereated text

"""

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

class Model():
    '''
       rnn_size is a list where each element is hidden_size of a layer
    '''
    def __init__(self, batch_size, seq_len, vocab_size, rnn_size=[128, 128], num_layers=2, training=True, use_dropout=True, lr=0.001, embed_size=10, dropout_rate=0.5):
        self.batch_size =  batch_size
        self.seq_len = seq_len
        if training == False:
            self.batch_size = 1
            self.seq_len = 1
        
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.training = training
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.embed_size = embed_size

        tf.reset_default_graph()
        self.X = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_len])
        self.y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_len])
        self.predict, self.loss = self.model(self.X, self.y)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()
    '''
      _input: (batch_size, seq_len)
      @return: (batch_size, seq_len, embed_size)
    '''
    def embed_layer(self, _input):
        embed_matrix = tf.get_variable('embedding', shape=[self.vocab_size, self.embed_size], dtype=tf.float32)
        embed_input = tf.nn.embedding_lookup(embed_matrix, _input)
        if self.training and self.use_dropout:
            embed_inp_dropout = tf.nn.dropout(embed_input, keep_prob = 1 - self.dropout_rate)
            return embed_inp_dropout
        return embed_input
    
    
    def rnn_cell(self, layer_num):
        return tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size[layer_num])
    
    def multi_rnn(self):
        cell = []
        for layer_i in range(self.num_layers):
            if self.training and self.use_dropout:
                cell.append(tf.contrib.rnn.DropoutWrapper(self.rnn_cell(layer_i), output_keep_prob=1.0-self.dropout_rate))
                continue
            cell.append(self.rnn_cell(layer_i))
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell)
        return multi_rnn_cell
    
    '''
       embed_inp: (batch_size, seq_len, embed_size)
       @return: (batch_size, seq_len, hidden_size)
    '''
    def Recurrent_layers(self, embed_inp):
        multi_rnn_cell = self.multi_rnn()
        self.initial_state = multi_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(multi_rnn_cell, embed_inp, initial_state=self.initial_state)
        return outputs, state
    
    '''
       @return: (batch_size * seq_len, vocab_size) x 2
    '''
    def Output_layers(self, outputs, hidden_size):
        flatten_out = tf.reshape(outputs, shape=[-1, hidden_size]) # shape: batch_size * seq_len, hidden_size
        softmax_W = tf.get_variable(name="out_W", shape=[hidden_size, self.vocab_size], dtype=tf.float32, initializer=tf.initializers.variance_scaling)
        softmax_b = tf.get_variable(name="out_b", shape=[self.vocab_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        logits = tf.matmul(flatten_out, softmax_W) + softmax_b
        prob = tf.nn.softmax(logits)
        return logits, prob
    
    def compute_loss(self, logits, y):
        targets = tf.reshape(y, shape=[-1])
        mean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
        return mean_loss
    
    def model(self, X, y):
        embed_inp = self.embed_layer(X)
        rnn_out, self.final_state = self.Recurrent_layers(embed_inp)
        logits, self.prob = self.Output_layers(rnn_out, self.rnn_size[-1]) # logits(batch_size * seq_len, vocab_size)
        loss = self.compute_loss(logits, y)
        pred = tf.reshape(tf.argmax(logits, axis=1), [self.batch_size, self.seq_len])
        return pred, loss
    
        
    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        with self.session as sess:
            sess.run(init)
            step = 0    
            new_state = sess.run(self.initial_state)
            for X_batch, y_batch in batch_generator:
                 step += 1
                 start = time.time()
                 new_state, loss_val, _ = sess.run([self.final_state, self.loss, self.train_op], feed_dict={self.X: X_batch, self.y: y_batch, self.initial_state: new_state})
                 end = time.time()
                 if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(loss_val),
                          '{:.4f} sec/batch'.format((end - start)))
                 if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                 if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
            
    def sample(self, char, vocab, n, start_string):
         sess = self.session 
         samples = list(start_string)
         new_state = sess.run(self.initial_state)
         pred = np.ones((self.vocab_size, ))
         for ch in start_string:
             x = np.zeros((1,1))
             x[0,0] = vocab[ch]
             pred, new_state = sess.run([self.prob, self.final_state],
                                        feed_dict={self.X: x, self.initial_state: new_state})
         ch = char[pick_top_n(pred, self.vocab_size)]
         samples.append(ch)
         
         for i in range(1, n):
             x = np.zeros((1,1))
             x[0,0] = vocab[ch]
             pred, new_state = sess.run([self.prob, self.final_state],
                                        feed_dict={self.X: x, self.initial_state: new_state})
             ch = char[pick_top_n(pred, self.vocab_size)]
             samples.append(ch)
         return "".join(samples)
     
    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))