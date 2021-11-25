# ##############################################################################
#
# Version: 0.0.1 (24 November 2021)
# Author: Benjamín Ramírez (benjaminramirezg@gmail.com)
#
# ##############################################################################

"""
Library with classes to train text classification models and predict with them
"""

import os
import re
import json
import pickle
import string
import logging
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from nltk.tokenize import TweetTokenizer
from tensorflow.keras.preprocessing import text

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dense, Activation, Dropout

log = logging.getLogger('models')

_HYPERPARAMETERS_DESCRIPTION = {
    'num_words': int,
    'hidden_layer_size': int,
    'dropout': float,
    'train_percent': float,
    'batch_size': int,
    'epochs': int
}

_TEXT_FIELD_NAME = 'text'
_LABEL_FIELD_NAME = 'label'

_MODEL_FILE_NAME = 'model.h5'
_TOKENIZER_FILE_NAME = 'tokenizer.pkl'
_NORMALIZER_FILE_NAME = 'normalizer.pkl'
_VECTORIZER_FILE_NAME = 'vectorizer.pkl'

class TextNormalizer(object):
    """
    Define objects intended to normalize text
    """
    def __init__(self, normalize_unicode=None, remove_break_lines=None):
        """
        Initialize the object with specific parameters
        """
        self._normalize_unicode = normalize_unicode
        self._remove_break_lines = remove_break_lines

    def normalize(self, text):
        """
        Normalize text passed as argument
        """
        if self._normalize_unicode:
            try:
                encoded_text = text.encode('unicode-escape')
                encoded_text = encoded_text.replace(b'\\\\u', b'\\u')
                decoded_text = encoded_text.decode('unicode-escape')
                text = decoded_text.encode('utf-8', 'ignore').decode('utf-8')
            except UnicodeEncodeError:
                log.info('[ Warning ] Unable to encode Unicode characters')
            except UnicodeDecodeError:
                log.info('[ Warning ] Unable to dencode Unicode characters')

        if self._normalize_unicode:
            text = text.replace("\n", " ")

        return text

class TextTokenizer(object):
    """
    Define objects intended to tokenize text
    """
    def __init__(self, filter_punctuation=None, filters_regex=None, lower=None):
        """
        Initialize the object with specific parameters
        """
        self._tokenizer = TweetTokenizer()
        self._filter_punctuation = filter_punctuation
        self._lower = lower
        if filters_regex is None:
            filters_regex = []
        elif isinstance(filters_regex, str):
            filters_regex = [filters_regex]
        elif not isinstance(filters_regex, list):
            raise ValueError('Value for parameter filters_regex must be a string or an array')
        self._filters_regex = filters_regex

    def tokenize(self, text):
        """
        Tokenize text passed as argument
        """
        tokens = self._tokenizer.tokenize(text)

        if self._filter_punctuation:
            tokens = list(filter(lambda token: token not in string.punctuation, tokens))
        for filter_regex in self._filters_regex:
            tokens = list(filter(lambda token: not re.match(filter_regex, token), tokens))
        if self._lower:
            tokens = [token.lower() for token in tokens]

        return tokens

class BinaryTextClassifierTrainer(object):
    """
    Define objects to train binary text classifiers (1/0)
    """

    def __init__(self, normalizer=None, tokenizer=None, config_filepath=None):
        """
        Initialize the object with specific utilities to
        preprocess text
        """
        self._normalizer = normalizer
        self._tokenizer = tokenizer
        self._default_hyperparameters = {}
        self._default_hyperparameters = self._get_default_hyperparameters(
            config_filepath
            )

    def _get_default_hyperparameters(self, filepath):
        """
        Read default hyperparameters from config file
        """
        file_content = None
        try:
            filehandle = open(filepath, 'r', encoding='utf-8')
            file_content = filehandle.read()
            filehandle.close()
        except OSError:
            raise OSError(
        'Unable to open config file {}'.format(filepath)
        )

        hyperparameters = None
        try:
            hyperparameters = json.loads(file_content)
        except ValueError:
            raise ValueError(
        'Unable to parse json config file {}'.format(filepath)
        )

        for hyperparameter, type_ in _HYPERPARAMETERS_DESCRIPTION.items():
            if hyperparameter not in hyperparameters:
                raise AttributeError(
            'Mandatory hyperparameter {} not provided'.format(hyperparameter)
            )
            value = hyperparameters[hyperparameter]
            if not isinstance(value, type_):
                raise ValueError(
            'Wrong value {} provided for hyperparameter {}'.format(str(value), hyperparameter)
            )

        return hyperparameters

    def _get_hyperparameters(self, hyperparameters):
        """
        Return hyperparameter values if provided. Otherwise default
        values are provided
        """
        if not hyperparameters:
            hyperparameters = self._default_hyperparameters
        else:
            for hyperparameter, type_ in _HYPERPARAMETERS_DESCRIPTION.items():
                if hyperparameter not in hyperparameters:
                    hyperparameters[hyperparameter] = self._default_hyperparameters[hyperparameter]
                else:
                    value = hyperparameters[hyperparameter]
                    if not isinstance(value, type_):
                        raise ValueError(
                    'Wrong value {} provided for hyperparameter {}'.format(str(value), hyperparameter)
                    )
        return hyperparameters

    def _create_model(self, hyperparameters=None):
        """
        Create a model to be trained
        """
        num_words = hyperparameters['num_words']
        hidden_layer_size = hyperparameters['hidden_layer_size']
        dropout = hyperparameters['dropout']

        model = Sequential()
        model.add(Dense(
            hidden_layer_size, input_shape=(num_words,)
            ))
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(
            loss="binary_crossentropy", optimizer="adam",
            metrics=["accuracy", Precision(), Recall()]
            )

        model.summary()

        return model

    def _save_model(self, model=None, vectorizer=None, output_folder=None):
        """
        Save model artifacts
        """

        model_path = os.path.join(output_folder, _MODEL_FILE_NAME)
        vectorizer_path = os.path.join(output_folder, _VECTORIZER_FILE_NAME)
        normalizer_path = os.path.join(output_folder, _NORMALIZER_FILE_NAME)
        tokenizer_path = os.path.join(output_folder, _TOKENIZER_FILE_NAME)

        artifacts = [
            [vectorizer, vectorizer_path],
            [self._normalizer, normalizer_path],
            [self._tokenizer, tokenizer_path]
        ]

        for artifact_info in artifacts:
            artifact_object = artifact_info[0]
            artifact_path = artifact_info[1]
            pickle.dump(
                artifact_object, open(artifact_path,'wb+'),
                protocol=pickle.HIGHEST_PROTOCOL
            )

        model.save(model_path)

    def _evaluate_model(self, model=None, x=None, y=None):
        """
        Evaluate model
        """
        evaluation = model.evaluate(x, y)
        metrics = {
            'loss': evaluation[0], 'accuracy': evaluation[1],
            'precision': evaluation[2], 'recall': evaluation[3]
        }
        return metrics

    def _split_train_test_texts(self, dataset=None, train_percent=None):
        """Splits dataset into train and evaluation dataset"""

        dataset = shuffle(dataset, random_state=0)
        train_size = int(len(dataset) * train_percent)
        train_texts = dataset[:train_size]
        test_texts = dataset[train_size:]
        return train_texts, test_texts

    def train(self, dataset=None, hyperparameters=None, output_folder=None):
        """
        Trains a model given training data and a specific
        set of hyperparameters
        """
        if not dataset:
            raise AttributeError('Not dataset provided to train')

        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('Dataset to train must be a DataFrame')

        if _TEXT_FIELD_NAME not in dataset.columns:
            raise AttributeError(
        'No mandatory column {} in dataset'. format(_TEXT_FIELD_NAME)
        )

        if _LABEL_FIELD_NAME not in dataset.columns:
            raise AttributeError(
        'No mandatory column {} in dataset'. format(_LABEL_FIELD_NAME)
        )

        if hyperparameters and not isinstance(hyperparameters, dict):
            raise ValueError('Hyperparameters must be a dictionary')

        if output_folder and not isinstance(output_folder, str):
            raise ValueError('Output folder must be a string')

        if not os.path.isdir(output_folder):
            try:
                os.makedirs(output_folder)
            except OSError:
                raise OSError('Unable to create output folder {}')

        hyperparameters = self._get_hyperparameters(hyperparameters)

        num_words = hyperparameters['num_words'] 
        batch_size = hyperparameters['batch_size']
        epochs = hyperparameters['epochs']
        train_percent = hyperparameters['train_percent']

        train_dataset, eval_dataset = self._split_train_test_texts(
            dataset=dataset, train_percent=train_percent
        )

        train_texts = train_dataset[_TEXT_FIELD_NAME].to_list()
        train_texts = [self._normalizer.normalizer(text) for text in train_texts]
        train_texts = [self._tokenizer.tokenizer(text) for text in train_texts]
        eval_texts = eval_dataset[_TEXT_FIELD_NAME].to_list()
        eval_texts = [self._normalizer.normalizer(text) for text in eval_texts]
        eval_texts = [self._tokenizer.tokenizer(text) for text in eval_texts]
        train_labels = train_dataset[_LABEL_FIELD_NAME].to_list()
        eval_labels = eval_dataset[_LABEL_FIELD_NAME].to_list()

        vectorizer = text.Tokenizer(num_words=num_words)
        vectorizer.fit_on_texts(train_texts)
        x_train = vectorizer.texts_to_matrix(train_texts, mode="tfidf")
        y_train = np.array(train_labels)
        x_eval = vectorizer.texts_to_matrix(eval_texts, mode="tfidf")
        y_eval = np.array(eval_labels)

        model = self._create_model(hyperparameters=hyperparameters)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
        self._save_model(model=model, vectorizer=vectorizer, output_folder=output_folder)
        metrics = self._evaluate_model(model=model, x=x_eval, y=y_eval)

        return metrics

class BinaryTextClassifier(object):
    """
    Define objects to load a binary text classifiers model and predict with it
    """

    def __init__(self, artifacts_folder=None):
        """
        Loads model from serialized objctes in artifacts_folder
        """
        if not os.path.isdir(artifacts_folder):
            raise ValueError('Artifacts folder {} not found'.format(artifacts_folder))

        model_path = os.path.join(artifacts_folder, _MODEL_FILE_NAME)
        vectorizer_path = os.path.join(artifacts_folder, _VECTORIZER_FILE_NAME)
        normalizer_path = os.path.join(artifacts_folder, _NORMALIZER_FILE_NAME)
        tokenizer_path = os.path.join(artifacts_folder, _TOKENIZER_FILE_NAME)

        if not os.path.isfile(model_path):
            raise ValueError('File for model {} not found'.format(model_path))
        if not os.path.isfile(vectorizer_path):
            raise ValueError('File for vectorizer {} not found'.format(vectorizer_path))
        if not os.path.isfile(tokenizer_path):
            raise ValueError('File for tokenizer {} not found'.format(tokenizer_path))
        if not os.path.isfile(normalizer_path):
            raise ValueError('File for normalizer {} not found'.format(normalizer_path))

        self._normalizer = self._load_pickle(normalizer_path)
        self._tokenizer = self._load_pickle(tokenizer_path)
        self._vectorizer = self._load_pickle(vectorizer_path)
        self._model = load_model(model_path)

    def _load_pickle(self, path):
        """Loads pickle serialized object"""
        artifact = None
        with open(path, 'rb') as filehandle:
            artifact = pickle.load(filehandle)
        return artifact

    def predict(self, texts):
        """
        Takes a text and returns a value between 1 and 0
        according to the loaded model
        """
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValueError('Texts argument must be a string or a list of strings')

        texts = [self._normalizer.normalize(text) for text in texts]
        texts = [self._tokenizer.tokenize(text) for text in texts]
        x = [self._vectorizer.tokenize(text) for text in texts]

        predictions = self._model.predict(x)
        if len(predictions) == 1:
            predictions = predictions[0]

        return predictions
