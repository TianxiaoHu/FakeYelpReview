# -*- coding: utf-8 -*-
from tqdm import tqdm
import argparse
import pandas as pd


def parse_and_save(input, output, review_num=None):
    tqdm.pandas()
    if review_num:
        data = pd.read_csv(input, usecols=["text", "stars"], dtype={
            "text": str, "stars": str}, nrows=int(review_num))
    else:
        data = pd.read_csv(input, usecols=["text", "stars"], dtype={
            "text": str, "stars": str})

    def filter_non_eng_char(string):
        return "".join(list(filter(lambda x: ord(x) < 128, string)))

    def convert_to_token(star):
        token_dict = {"1.0": "<ONE>", "2.0": "<TWO>",
                      "3.0": "<THREE>", "4.0": "<FOUR>", "5.0": "<FIVE>"}
        return token_dict[star]

    data = data.dropna()
    print("Converting... step 1/2..", '\n')
    data.text = data.text.progress_apply(filter_non_eng_char)
    print("Converting... step 2/2..", '\n')
    data.stars = data.stars.progress_apply(convert_to_token)
    data["converted_review"] = data.stars + data.text

    with open(output, "w") as f:
        print("Writing...", '\n')
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            f.write(row.converted_review + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate char-level input from csv file for RNN.',
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default='./dataset/yelp_dataset/review.csv',
        help='The csv file to convert.',
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Data saving path.'
    )

    parser.add_argument(
        '-n', '--review_num',
        type=str,
        default=None,
        help='How many reviews to convert.',
    )
    args = parser.parse_args()

    csv_file = args.input
    output_file = args.output
    review_num = args.review_num
    parse_and_save(csv_file, output_file, review_num)
