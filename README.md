# FakeYelpReview
Final project for CS282 - Designing, Visualizing and Understanding Deep Neural Networks (Spring 2019) @ UC Berkeley. 

Data downloaded from [Yelp Open Dataset](https://www.yelp.com/dataset).

We have 4 models implemented in **4 branch**. Check project branch for more details.

branch master: character-level vanilla two-layer LSTM

branch textgenrnn: character-level LSTM with Attention Layer

branch wordlevel: word-level generation using GRU

branch gpt-2: state-of-the-art GPT-2 model

Refer to our poster and report for more details.

## Data Preprocessing

### Convert `json` to `csv` format

Original data is saved at `FakeYelpReview/dataset/`

After decompression, there are **7** json files under `FakeYelpReview/dataset/yelp_dataset`. However, we only use `review.json` and `business.json`for training and generation.

```bash
# convert json to csv
# json_to_csv_converter.py downloaded from https://github.com/Yelp/dataset-examples
# WARNING: the last update is in 2014, so use python 2.7 to execute the python script
# converted csv file will be saved at `FakeYelpReview/dataset/`
python json_to_csv_converter.py ./dataset/yelp_dataset/review.json

# updated script json_to_csv_converter_py3.py to support python 3
python json_to_csv_converter_py3.py ./dataset/yelp_dataset/business.json

# all kinds of reviews are contained in review.csv
# only extract reviews for restaurants
# converted file saved at `FakeYelpReview/dataset/yelp_dataset`
python extract_restaurant_review.py
# Total review number: 6685900
# Restaurant review number: 4201684
```

### Prepare Input

Remove non-English chars and remove line break symbols(`\n`).

```bash
# generate a tiny input dataset (~137K) for sanity check
python generate_char_level_input.py -o dataset/input_tiny.txt -n 200

# generate a small input dataset (~5.7M) for network tuning
python generate_char_level_input.py -o dataset/input_small.txt -n 10000

# generate a small input dataset (~4.7M) for 5 star reviews
python generate_char_level_input.py -o dataset/input_small_5s.txt -n 10000 -s 5
```

## Installation - for master branch

python version == 3.6

tensorflow==1.10.0

Keras==2.2.4

tqdm==4.31.1

## Model Training - for master branch

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

- training 10000 reivews on Google colab takes ~10min/epoch, accelerated with GPU. memory usage: ~15G
- training 20000 reviews on AWS p2.xlarge takes ~40min/epoch, accelrated with a K80. memory usage: ~30G

## Generated Sample - for master branch

**1-star**: Will not come here again. We were disappointed that I was the only ones thatthe hostess stopped here. Because of the Kateron that there was a bad serviceand the food was bland and overcooked.

**5-star**: This is the best in Phoenix area! The service was amazing. It was nice andtasty and creative in a large group of friendly and attentive staff and no otheromelet over the exception. I also order the potato, chicken and the chickenparmbialese.

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