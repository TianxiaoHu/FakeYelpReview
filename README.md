# FakeYelpReview - GPT-2 branch
GPT-2 is the state-of-art model, and we use it as the baseline model for word-level text generation. 

GPT-2 is a transformer-based language generation model, which is proposed by OpenAI in 2019. 

Model structure of simplified GPT-2: 12 repeated transformer blocks with multi-head attention structure. 

We fine-tuned the GPT-2 model using 1,000 reviews, which is enough to generate realistic comments. 

From the experimental result,  the performance of our model (Word-level Generation) is almost close to GPT-2, which proves the accuracy and reliability of our model.

## Installation

python version == 3.6

tensorflow==1.10.0

gpt-2-simple==0.3.1


## Model Training

```bash

# Fine-tune the GPT-2 using review_dataset.txt
# Running gpt_2_simple finetune review_dataset.txt will generate checkpoints for fine-tuned model.
# The pretrained model is 117M, and the finetuned model is 500 MB!
# temperature: 1.0
# batch size: 20
# epoch: 500
# model checkpoints and log saved per epoch in new folder under `checkpoint/run1`

pip3 install gpt_2_simple

# For finetuning (which will also download the model if not present):
! gpt_2_simple finetune review_dataset.txt

# For generation, which generates texts to files in a gen folder:
! gpt_2_simple generate
                       
```

## Generated sample

**1-star**: Worst dim sum place that I have ever been. They have black hair in their food and urine. The hostess had the whole restaurant and all of its personnel in front of her at the same time. It was incredibly rude and graphiced everything we did and where we were served.

**5-star**: We like to think that this is the best Thai restaurant in the area and possibly in Mississauga as far as Thai is concerned.  The food is certainly well prepared and the staff is extremely friendly. The ambience is calming and inviting. A traveler in need of a table?  This is a must try!

## Reference

https://github.com/minimaxir/gpt-2-simple


