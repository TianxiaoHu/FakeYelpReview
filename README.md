# FakeYelpReview
Final project for CS282 - Designing, Visualizing and Understanding Deep Neural Networks (Spring 2019) @ UC Berkeley. 

Data downloaded from [Yelp Open Dataset](https://www.yelp.com/dataset).

## Installation

TODO

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

### Prepare Generation Input

Remove non-English chars and remove line break symbols(`\n`).

```bash
# generate a tiny input dataset (~137K) for sanity check
python generate_char_level_input.py -o dataset/input_tiny.txt -n 200

# generate a small input dataset (~5.7M) for network tuning
python generate_char_level_input.py -o dataset/input_small.txt -n 10000

# generate a small input dataset (~4.7M) for 5 star reviews
python generate_char_level_input.py -o dataset/input_small_5s.txt -n 10000 -s 5
```

## Generation Model

### Word-level review generation

This part of our project utilizes word embedding to realize word-level review generation. Due to the large vocabulary size (typically tens of thousand) and limited computation resources, only 6000 most frequent words are selected. The embedded output is then fed as the input of a stacked two-layer gated recurrent units (GRU). Finally, the output is linearly projected to word space and yields a softmax probability.

To improve the model performance and training speed, within GRU, a dropout probability of 0.2 is employed to all reset and input gates. A dropout probability of 0.5 is employed between the word embedding layer and the GRU, and between the GRU and the fully-connected output layer. 

**Sample generated texts:**

- If you're looking for a small place to take a good time at the end of the night. I would recommend this place to anyone to try the <unk\>, and you won't be disappointed. Weeknight, and the food is great.
- My husband and I were looking for a little <unk\>. The pizza was delicious, the shrimp was good and the chicken was good! My friend had the chicken and the shrimp and it was delicious. The chips were great and the food was amazing!
- This is a great place to eat. I've had a few people here. The place is very nice and the service is exceptional. I would recommend the food and food.
- Very good, the food was delicious. I had a great meal and the food was great. I'm so glad I had the same thing that I had. I'm not sure if I was going to get a drink.
- A few years ago, and I was really excited to try for my first time. I'm glad I would be going back to the place. The food was good, the food was good, but the food was pretty good.
- I was told that the other employee came out and said they were. We sat in the front desk for our table. They brought us the food and the waiter was very friendly and nice.
 
 (Nothing changed except for formatting)
 
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
