# ##############################################################################
#
# Version: 0.0.1 (29 November 2021)
# Author: Benjamín Ramírez (benjaminramirezg@gmail.com)
#
# ##############################################################################

"""
Script to load a model and predict over text according to the model

Input texts must be received as a CSV dataset
Predictions are retrieved as a CSV dataset
"""

import argparse
import pandas as pd
from models import BinaryTextClassifier

# Parsing input parameters
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Path to the CSV file with the input dataset", required=True)
parser.add_argument("-s", "--separator", help="Character used as field delimiter in CSV", default=',')
parser.add_argument("-t", "--texts", help="Name of the column of the CSV where text appears", default='text')
parser.add_argument("-l", "--labels", help="Name of the column of the CSV where labels appear", default='label')
parser.add_argument("-m", "--model", help="Path to the folder where the model to be used is saved", required=True)
parser.add_argument("-o", "--output", help="Path to the CSV file where predictions will be saved", required=True)
args = parser.parse_args()

# Loading model
print('Loading model from {}'.format(args.model))
model = BinaryTextClassifier(artifacts_folder=args.model)

# Reading dataset
print('Reading dataset {}'.format(args.dataset))
dataset = pd.read_csv(args.dataset, sep=args.separator)

if args.texts not in dataset.columns:
    raise AttributeError('No column {} found in the dataset'.format(args.texts))
texts = dataset[args.texts].to_list()

# Predicting over text in dataset
print('Predicting over text in dataset')
predictions = model.predict(texts)

# Saving predictions in new dataset
print('Saving predictions in new dataset {}'.format(args.output))
dataset[args.labels] = predictions
dataset.to_csv(args.output, sep=args.separator, index=False)
