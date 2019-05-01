# -*- coding: utf-8 -*-
from tqdm import tqdm
import argparse
import pandas as pd


def parse_and_save(input, output, review_num=None, star=0):
    tqdm.pandas()
    if review_num:
        # read 7 times of review_num to ensure enough reviews can be covered
        data = pd.read_csv(input, usecols=["text", "stars"], dtype={
            "text": str, "stars": int}, nrows=int(10 * review_num))
    else:
        data = pd.read_csv(input, usecols=["text", "stars"], dtype={
            "text": str, "stars": int})

    def filter_non_eng_char_and_line_break(string):
        eng_str = "".join(list(filter(lambda x: ord(x) < 128, string)))
        return eng_str.replace("\n", "")

    data = data.dropna()
    print("Converting... step 1/2..", '\n')
    data.text = data.text.progress_apply(filter_non_eng_char_and_line_break)
    print("Filtering... step 2/2..", '\n')
    if star:
        data = data[data.stars == star]
    write_num = 0

    with open(output, "w") as f:
        print("Writing...", '\n')
        for index, row in tqdm(data.iterrows(), total=min(data.shape[0], review_num)):
            f.write(row.text + '\n')
            write_num += 1
            if review_num and write_num >= review_num:
                break

    print(write_num, "reviews write to", output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate char-level input from csv file for RNN.',
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default='./dataset/yelp_dataset/restaurant_review.csv',
        help='The csv file to convert.',
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Data saving path.'
    )

    parser.add_argument(
        '-n', '--review_num',
        type=int,
        default=None,
        help='How many reviews to convert.',
    )

    parser.add_argument(
        '-s', '--stars',
        type=int,
        default=0,
        help='Only extract n-star reviews.',
    )
    args = parser.parse_args()

    csv_file = args.input
    output_file = args.output
    review_num = args.review_num
    star_num = args.stars
    parse_and_save(csv_file, output_file, review_num, star_num)
