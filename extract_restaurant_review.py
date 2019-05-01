# -*- coding: utf-8 -*-
"""
Only keep reviews from restaurants.

New `restaurant_review.csv` saved to the same path.
"""
from __future__ import print_function
import os
import argparse
import pandas as pd


def main(input, output):
    df_business = pd.read_csv(os.path.join(input, 'business.csv'))
    df_business = df_business[["business_id", "categories"]]
    df_business.fillna(inplace=True, value="")
    restaurants = df_business[df_business['categories'].str.contains(
        'Restaurants')]

    df_review = pd.read_csv(os.path.join(input, 'review.csv'))
    print("Total review number:", df_review.shape[0])

    restaurant_review = pd.merge(restaurants, df_review, on='business_id')
    print("Restaurant review number:", restaurant_review.shape[0])
    restaurant_review.to_csv(os.path.join(output, 'restaurant_review.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, default='./dataset/yelp_dataset',
                        help='Input folder: business.csv and review.csv')

    parser.add_argument('-o', '--output', type=str, default='./dataset/yelp_dataset',
                        help='Output folder: restaurant_review.csv')

    args = parser.parse_args()

    input, output = args.input, args.output

    main(input, output)
