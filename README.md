# FakeYelpReview - textgenrnn branch
Textgenrnn attention-based char-by-char model that we used to compare with vanilla LSTM model.
Model structure: two 256 cell LSTMs followed by an attention layer with embedding size 100.
Batch size is 1024, and 30 tokens are considered before predicting the next one.

## Installation

python version == 3.6

tensorflow==1.10.0

Keras==2.2.4

textgenrnn (can be installed with pip3)


## Model Training


```bash
# train two-layer char level LSTM model with 256 cells each and 100 embedding size.
# Input should be text file and separate reviews in lines
# Running the script textgenrnn_training.py will generate three files during training (with the output name you defined): _weights.hdf5 file, _vocab.json, and _config.json
# Running the script will also automatically generate temperature 0.2 and 0.5 reviews after training and save in separate files
# Recommendation: Loss < 1
# input example text file name: input_5s.txt
# output model example name: text_model
# epoch: 20, generate reviews for each 5 epochs
# Numbers of reviews generated after training: 10
# Length of reviews generated after training 300

python textgenrnn_training.py -i input_5s.txt -o test_model -epo 20 -epo_gen 5 -n 10 -l 300

# Loading model locally with Python3:
# For more details, please see reference
from textgenrnn import textgenrnn
textgen = textgenrnn(weights_path='test_model_weights.hdf5',
                       vocab_path='test_model_vocab.json',
                       config_path='test_model_config.json')
                       
# Continue training process with data trainin_data.txt
textgen.train_from_file('training_data.txt', num_epochs=1, new_model=False)
                       
# Generate reviews (to file textgenrnn_texts.txt)
textgen.generate_samples(max_gen_length=300,n=10,temperature=0.2)
textgen.generate_to_file('textgenrnn_texts.txt', max_gen_length=300, n=10, temperature=0.2)
                       
```

## Generated sample

**1-star**: The food was okay. The rice was ok, but the waitress was pretty stupid for the first time. We were told they were out of bread. The sandwich was so tough they didn't have a couple of salads.

**5-star**: I love this place.  I have been going to this location and was very convenient. The food was amazing and the service is always good. The owner is super friendly and helpful. The service was fast.

## Reference

https://github.com/minimaxir/textgenrnn

