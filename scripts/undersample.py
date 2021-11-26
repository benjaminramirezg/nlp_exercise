# ##############################################################################
#
# Version: 0.0.1 (26 November 2021)
# Author: Benjamín Ramírez (benjaminramirezg@gmail.com)
#
# ##############################################################################

"""
Script to balance datasets for binary classifiers
by undersampling the most populated category
"""

import argparse
import pandas as pd
from sklearn.utils import shuffle

# Parsing input parameters
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Path to the CSV file with the dataset", required=True)
parser.add_argument("-s", "--separator", help="Character used as field delimiter in CSV", default=',')
parser.add_argument("-t", "--texts", help="Name of the column of the CSV where text appears", default='text')
parser.add_argument("-l", "--labels", help="Name of the column of the CSV where labels appear", default='label')
parser.add_argument("-o", "--output", help="Path to the output version of the dataset", required=True)
args = parser.parse_args()

# Reading dataset
print('Reading dataset {}'.format(args.dataset))
dataset = pd.read_csv(args.dataset, sep=args.separator)

# Spliting datasets by classes
positive_dataset = dataset[dataset[args.labels]==1]
negative_dataset = dataset[dataset[args.labels]==0]

large_dataset = None
small_dataset = None

if len(negative_dataset) > len(positive_dataset):
    large_dataset = negative_dataset
    small_dataset = positive_dataset
else:
    large_dataset = positive_dataset
    small_dataset = negative_dataset

# Undersampling the large class of the dataset
print('Undersampling larger class')
large_dataset = shuffle(large_dataset, random_state=0)
undersampled_dataset = large_dataset[:len(small_dataset)]

# Join the resulting datasets
resulting_dataset = pd.concat([undersampled_dataset, small_dataset])
resulting_dataset = shuffle(resulting_dataset, random_state=0)

# Saving the undersampled dataset
print('Saving balanced version in {}'.format(args.output))
resulting_dataset.to_csv(args.output, sep=args.separator, index=False)