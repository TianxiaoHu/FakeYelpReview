from textgenrnn import textgenrnn

import argparse
#usage:
#python textgenrnn_training.py -i input_5s.txt -o test_model -epo 1 -epo_gen 1

def train_textgenrnn(input, output, model_cfg,train_cfg):
    textgen = textgenrnn(name=output)

    train_function = textgen.train_from_file

    train_function(
        file_path=input,
        new_model=True,
        num_epochs=train_cfg['num_epochs'],
        gen_epochs=train_cfg['gen_epochs'],
        batch_size=1024,
        train_size=train_cfg['train_size'],
        dropout=train_cfg['dropout'],
        validation=train_cfg['validation'],
        is_csv=train_cfg['is_csv'],
        rnn_layers=model_cfg['rnn_layers'],
        rnn_size=model_cfg['rnn_size'],
        rnn_bidirectional=model_cfg['rnn_bidirectional'],
        max_length=model_cfg['max_length'],
        dim_embeddings=100,
        word_level=model_cfg['word_level'])


def generate_review(num,text_len,output):
    textgen = textgenrnn(weights_path=output+'_weights.hdf5',
                       vocab_path=output+'_vocab.json',
                       config_path=output+'_config.json')
    textgen.generate_to_file(output+'_temp2_textgenrnn_texts.txt', max_gen_length=text_len,n=num,temperature=0.2)
    textgen.generate_to_file(output+'_temp5_textgenrnn_texts.txt', max_gen_length=text_len,n=num,temperature=0.)


if __name__ == '__main__':

    #for training
    parser = argparse.ArgumentParser(
        description='Train char-level textgenrnn model.',
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default='./dataset/input_5s.txt',
        help='The text file (one review per line).',
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='model_output',
        help='Output model name.'
    )

    
    parser.add_argument(
        '-epo', '--epoch',
        type=int,
        default=1,
        help='How many epoch to train',
    )
    parser.add_argument(
        '-epo_gen', '--epoch_generation',
        type=int,
        default=1,
        help='Generate review on screen after how many epoch of training',
    )
    
    #for generating
    parser.add_argument(
        '-n', '--review_num',
        type=int,
        default=10,
        help='How many reviews to generate.',
    )
    
    parser.add_argument(
        '-l', '--text_len',
        type=int,
        default=300,
        help='How long is the review generated',
    )
    
    args = parser.parse_args()
    
    model_cfg = {
    'word_level': False,   # set to True if want to train a word-level model (requires more data and smaller max_length)
    'rnn_size': 256,   # number of LSTM cells of each layer (128/256 recommended)
    'rnn_layers': 2,   # number of LSTM layers (>=2 recommended)
    'rnn_bidirectional': False,   # consider text both forwards and backward, can give a training boost
    'max_length': 30,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_words': 10000,   # maximum number of words to model; the rest will be ignored (word-level model only)
    }

    train_cfg = {
    'line_delimited': True,   # set to True if each text has its own line in the source file
    'num_epochs': args.epoch,   # set higher to train the model for longer
    'gen_epochs': args.epoch_generation,   # generates sample text from model after given number of epochs
    'train_size': 1,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'dropout': 0.0,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'validation': False,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
    'is_csv': False   # set to True if file is a CSV exported from Excel/BigQuery/pandas
    }

    
    train_textgenrnn(args.input, args.output, model_cfg, train_cfg)
    generate_review(args.review_num, args.text_len, args.output)

