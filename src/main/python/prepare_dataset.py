# ##############################################################################
#
# Version: 0.0.1 (29 November 2021)
# Author: Benjamín Ramírez (benjaminramirezg@gmail.com)
#
# ##############################################################################

"""
Script to prepare the dataset suitable to train and evaluate a model

It takes as input an original CSV dataset.
It undersamples the most populated category to get a balanced version of the dataset
It splits the resulting balanced dataset into training, validation and testing sets
It saves all resulting datasets in an output folder
"""

import os
import argparse
import pandas as pd
from sklearn.utils import shuffle

# Names of the files that will be created as output
_BALANCED_DATASET_FILE_NAME = 'balanced_dataset.csv'
_TRAINING_SET_FILE_NAME = 'training_set.csv'
_VALIDATION_SET_FILE_NAME = 'validation_set.csv'
_TESTING_SET_FILE_NAME = 'testing_set.csv'

def split_train_val_test(dataset=None, train_proportion=None):
    """
    Splits dataset into train and evaluation dataset

    :param dataset: DataFrame with the input dataset to be splitted
    :param train_proportion: float between 0 and 1 that defines the
                             proportion of data that will be destined
                             to training
    :return: three datasets of type DataFrame (train, validation, test)
    """

    dataset = shuffle(dataset, random_state=0)
    train_size = int(len(dataset) * train_proportion)
    train_texts = dataset[:train_size]
    testing_texts = dataset[train_size:]

    val_size = int(len(testing_texts) * 0.50)
    val_texts = testing_texts[:val_size]
    test_texts = testing_texts[val_size:]

    return train_texts, val_texts, test_texts

def undersample(dataset=None):
    """
    Creates balanced version of the dataset by undersampling
    the largest class
    :param dataset: DataFrame with the input dataset to be undersample
    :return: the undersampled dataset of type DataFrame
    """
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
    large_dataset = shuffle(large_dataset, random_state=0)
    undersampled_dataset = large_dataset[:len(small_dataset)]

    # Join the resulting datasets
    balanced_dataset = pd.concat([undersampled_dataset, small_dataset])
    balanced_dataset = shuffle(balanced_dataset, random_state=0)

    return balanced_dataset

if __name__ == '__main__':

    # Parsing input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to the CSV file with the input dataset", required=True)
    parser.add_argument("-s", "--separator", help="Character used as field delimiter in CSV", default=',')
    parser.add_argument("-t", "--texts", help="Name of the column of the CSV where text appears", default='text')
    parser.add_argument("-l", "--labels", help="Name of the column of the CSV where labels appear", default='label')
    parser.add_argument("-p", "--proportion", help="Proportion of dataset used to create de training set (value between 0 and 1)", default=0.80)
    parser.add_argument("-o", "--output", help="Path to a folder where datasets will be saved", required=True)
    args = parser.parse_args()

    # Creating output folder
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # Reading dataset
    print('Reading dataset {}'.format(args.dataset))
    dataset = pd.read_csv(args.dataset, sep=args.separator)

    # Getting balanced version of the dataset
    print('Undersampling largest class')
    balanced_dataset = undersample(dataset=dataset)

    # Saving the undersampled dataset
    balanced_dataset_path = os.path.join(args.output, _BALANCED_DATASET_FILE_NAME)
    print('Saving balanced dataset in {} (size {})'.format(balanced_dataset_path, str(len(balanced_dataset))))
    balanced_dataset.to_csv(balanced_dataset_path, sep=args.separator, index=False)

    # Splitting balanced dataset into training, validation and testing sets
    print('Splitting dataset into training, validation and testing sets')
    training_set, validation_set, testing_set = split_train_val_test(
        dataset=balanced_dataset, train_proportion=float(args.proportion)
        )

    # Saving training set
    training_set_path = os.path.join(args.output, _TRAINING_SET_FILE_NAME)
    print('Saving training set in {} (size {})'.format(training_set_path, str(len(training_set))))
    training_set.to_csv(training_set_path, sep=args.separator, index=False)

    # Saving validation set
    validation_set_path = os.path.join(args.output, _VALIDATION_SET_FILE_NAME)
    print('Saving validation set in {} (size {})'.format(validation_set_path, str(len(validation_set))))
    validation_set.to_csv(validation_set_path, sep=args.separator, index=False)

    # Saving testing set
    testing_set_path = os.path.join(args.output, _TESTING_SET_FILE_NAME)
    print('Saving testing set in {} (size {})'.format(testing_set_path, str(len(testing_set))))
    testing_set.to_csv(testing_set_path, sep=args.separator, index=False)
