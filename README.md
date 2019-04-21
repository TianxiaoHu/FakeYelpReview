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