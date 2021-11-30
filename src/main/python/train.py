# ##############################################################################
#
# Version: 0.0.1 (29 November 2021)
# Author: Benjamín Ramírez (benjaminramirezg@gmail.com)
#
# ##############################################################################

"""
Script to launch the training of a model of binary classification of text

A training dataset and a validation dataset must be provided for training
A configuration file must be provided to setup the model and the preprocessing
The resulting model will be saved in a folder and the metrics will be printed in STDOUT
"""

import json
import argparse
import pandas as pd
from models import ConfigManager
from models import TextTokenizer
from models import TextNormalizer
from models import BinaryTextClassifierTrainer

# Parsing input parameters
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Path to the CSV file with the training dataset", required=True)
parser.add_argument("-v", "--validation", help="Path to the CSV file with the validation dataset", required=True)
parser.add_argument("-s", "--separator", help="Character used as field delimiter in CSV", default=',')
parser.add_argument("-t", "--texts", help="Name of the column of the CSV where text appears", default='text')
parser.add_argument("-l", "--labels", help="Name of the column of the CSV where labels appear", default='label')
parser.add_argument("-c", "--config", help="Path to the JSON file with the config info", required=True)
parser.add_argument("-o", "--output", help="Path to the folder where artifacts will be saved", required=True)
args = parser.parse_args()

# Reading config parameters from file
config = ConfigManager(config_file=args.config)

# Creating objects to preprocess text and train model
tokenizer = TextTokenizer(
    filter_punctuation=config.get_('filter_punctuation'),
    regex_filters=config.get_('regex_filters'),
    lower=config.get_('lower'),
    stopwords_language=config.get_('stopwords_language')
)

normalizer = TextNormalizer(
    normalize_unicode=config.get_('normalize_unicode'),
    remove_break_lines=config.get_('remove_break_lines')
)

trainer = BinaryTextClassifierTrainer(
    normalizer=normalizer,
    tokenizer=tokenizer,
    default_hyperparameters=config.get_('hyperparameters')
)

# Reading dataset
print('Reading training dataset {}'.format(args.dataset))
training_dataset = pd.read_csv(args.dataset, sep=args.separator)

print('Reading validation dataset {}'.format(args.validation))
validation_dataset = pd.read_csv(args.validation, sep=args.separator)

# Training model
print('Training model')

metrics = trainer.train(
    training_set=training_dataset,
    validation_set=validation_dataset,
    text_field=args.texts,
    label_field=args.labels,
    output_folder=args.output
    )

print ('Model trained and saved in {}'.format(args.output))
print('Performance of the model: {}'.format(metrics))