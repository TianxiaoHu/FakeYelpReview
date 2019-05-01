# -*- coding: utf-8 -*-
'''
Load model and generate new fake reviews.
'''
from __future__ import print_function
import io
import os
import sys
import time
import random
import argparse
import numpy as np
from keras.models import load_model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(model, start, length, temperature, maxlen=40):
    # maxlen should equal to training script's maxlen

    sentence = start
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(sentence)

    for i in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]

        sentence = sentence + next_char
        if(len(sentence) > maxlen):
            sentence = sentence[- maxlen:]

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='Input data of model training.')
    parser.add_argument('-m', '--model_path', type=str,
                        help='Generate using which model.')
    parser.add_argument('-s', '--start', type=str, default="",
                        help='Given start for fake review.')
    parser.add_argument('-l', '--length', type=int, default=100,
                        help='Parameter: review length.')
    parser.add_argument('-t', '--temperature', type=float, default=1.0,
                        help='Parameter: emperature.')
    parser.add_argument('-n', '--number', type=int, default=1,
                        help='How many reviews need to be generated.')

    args = parser.parse_args()

    input, model_path = args.input, args.model_path

    with io.open(input, encoding='utf-8') as f:
        text = f.read()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    model = load_model(model_path)

    start, length, temperature, number = args.start, args.length, args.temperature, args.number
    for i in range(number):
        generate(model, start, length, temperature)
