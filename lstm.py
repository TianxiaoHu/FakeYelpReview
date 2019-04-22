# -*- coding: utf-8 -*-
'''
Modified from https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

    (Example script to generate text from Nietzsche's writings.)

    At least 20 epochs are required before the generated text
    starts sounding coherent.

    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.

    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function
import io
import os
import sys
import time
import random
import argparse
import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop


def train(input, output, name, continue_from, learning_rate, batch_size, epoch):

    with io.open(input, encoding='utf-8') as f:
        text = f.read()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # make directory for model and log
    if not continue_from:
        folder_name = name + \
            time.strftime("-%Y-%m-%d_%H:%M:%S", time.localtime())
        save_path = os.path.join(output, folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # build the model: two layer LSTM
    print('Build model...')
    model = Sequential()
    if continue_from:
        exp_name, model_from = continue_from.split('/')
        save_path = os.path.join(output, exp_name)
        model = load_model(os.path.join(output, continue_from))
    else:
        model.add(LSTM(1024, input_shape=(
            None, len(chars)), return_sequences=True))
        model.add(LSTM(1024, input_shape=(None, len(chars))))
        model.add(Dense(len(chars), activation='softmax'))

    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def on_epoch_end(epoch, _):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        # start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            # sentence = text[start_index: start_index + maxlen]
            sentence_list = ['<ONE>', '<THREE>', '<FIVE>']
            for sentence in sentence_list:
                generated = sentence
                print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                for i in range(100):
                    x_pred = np.zeros((1, maxlen, len(chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, char_indices[char]] = 1.

                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = sample(preds, diversity)
                    next_char = indices_char[next_index]

                    sentence = sentence[1:] + next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                print()

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    log_callback = CSVLogger(os.path.join(save_path, 'log.csv'),
                             separator=',', append=True)
    save_callback = ModelCheckpoint(os.path.join(save_path,
                                                 "weights.{epoch:d}-{loss:.2f}.hdf5"),
                                    monitor='loss', period=1)
    lr_callback = LearningRateScheduler(lambda x: learning_rate)

    model.fit(x, y, batch_size=batch_size, epochs=epoch,
              callbacks=[print_callback, log_callback, save_callback, lr_callback])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, default='./dataset/input_tiny.txt',
                        help='Input txt dataset')
    parser.add_argument('-o', '--output', type=str,
                        help='Model and log saving path')
    parser.add_argument('-c', '--continue_from', type=str, default=None,
                        help='Continue training from checkpoint.')
    parser.add_argument('-n', '--name', type=str,
                        default="", help='Experiment name.')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-e', '--epoch', type=int, default=10)

    args = parser.parse_args()

    input, output = args.input, args.output
    continue_from, exp_name = args.continue_from, args.name
    learning_rate, batch_size, epoch = args.learning_rate, args.batch_size, args.epoch
    train(input, output, exp_name, continue_from, learning_rate, batch_size, epoch)