# FakeYelpReview
Final project for CS282 - Designing, Visualizing and Understanding Deep Neural Networks (Spring 2019) @ UC Berkeley. 

Data downloaded from [Yelp Open Dataset](https://www.yelp.com/dataset).

## Installation

TODO

## Data Preprocessing

### Convert `json` to `csv` format

Original data is saved at `FakeYelpReview/dataset/`

After decompression, there are **7** json files under `FakeYelpReview/dataset/yelp_dataset`. However, we only use `review.json` for training and generation.

```bash
# convert json to csv
# json_to_csv_converter.py downloaded from https://github.com/Yelp/dataset-examples
# WARNING: the last update is in 2014, so use python 2.7 to execute the python script
# converted csv file will be saved at `FakeYelpReview/dataset/`
python json_to_csv_converter.py ./dataset/yelp_dataset/review.json
```

### Prepare Char-level Generation Input

```bash
# generate a tiny input dataset (~137K) for sanity check
python generate_char_level_input.py -o dataset/input_tiny.txt -n 200

# generate a small input dataset (~5.7M) for network tuning
python generate_char_level_input.py -o dataset/input_tiny.txt -n 10000
```

**📌4/21/2019**: Use `input_small.txt` for two-layer LSTM.

## Generation Model

### two-layer LSTM

```bash
# train a new two-layer LSTM model using input_small.txt
# experiment name: small_lstm
# learning rate: 0.001
# batch size: 1024
# epoch: 20
# model checkpoints and log saved per epoch in new folder under `model/`
python lstm.py -i dataset/input_small.txt -o model/ -n small_lstm -l 0.001 -b 1024 -e 20

# loading model from checkpoint and continue training
# starting from checkpoint `weights.2-2.81.hdf5`
# changed learning rate to 1e-4
python lstm.py -i dataset/input_tiny.txt -o model/ -n small_lstm -c 'small_lstm-2019-04-22_04:09:06/weights.2-2.81.hdf5' -l 0.0001 -b 1024 -e 10
```

## Reference

### OOTB Implementation

https://github.com/minimaxir/textgenrnn

https://github.com/minimaxir/gpt-2-simple

### Online Tutorial

https://www.tensorflow.org/tutorials/sequences/text_generation

https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

https://www.dlology.com/blog/how-to-generate-realistic-yelp-restaurant-reviews-with-keras/

https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb

https://medium.com/@enriqueav/update-automatic-song-lyrics-creator-with-word-embeddings-e30de94db8d1