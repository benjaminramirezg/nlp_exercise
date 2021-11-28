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
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dense, Activation, Dropout

log = logging.getLogger('models')

_HYPERPARAMETERS_DESCRIPTION = {
    'num_words': int,
    'n_hidden_layers': int,
    'hidden_layer_size': int,
    'dropout': float,
    'batch_size': int,
    'epochs': int
}

_MODEL_FILE_NAME = 'model.h5'
_TOKENIZER_FILE_NAME = 'tokenizer.pkl'
_NORMALIZER_FILE_NAME = 'normalizer.pkl'
_VECTORIZER_FILE_NAME = 'vectorizer.pkl'

_STOPWORDS_SUPPORTED_LANGUAGES = ['english']


class ConfigManager(object):
    """Defines objects to manage configuration"""

    def __init__(self, config_file=None):
        """Loads parameter values from fole"""
        parameters = {}
        with open(config_file, 'r') as filehandle:
            parameters = json.loads(filehandle.read())
        self._parameters = parameters

    def get_(self, param):
        """Returns the value in the config for a paramaters"""
        value = None
        if param in self._parameters:
            value = self._parameters[param]
        return value

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

        if self._remove_break_lines:
            text = text.replace("\n", " ")
            text = text.replace("\\n", " ")

        return text

class TextTokenizer(object):
    """
    Define objects intended to tokenize text
    """
    def __init__(
        self, filter_punctuation=None, regex_filters=None,
        lower=None, stopwords_language=None
        ):
        """
        Initialize the object with specific parameters
        """

        if regex_filters is None:
            regex_filters = []
        elif isinstance(regex_filters, str):
            regex_filters = [regex_filters]
        elif not isinstance(regex_filters, list):
            raise ValueError('Value for filters_regex must be string or array')

        stopwords = None
        if stopwords_language is None:
            stopwords = []
        elif not isinstance(stopwords_language, str):
            raise ValueError(
        'Value for stopwords_language must be a string'
        )
        elif stopwords_language not in _STOPWORDS_SUPPORTED_LANGUAGES:
            raise ValueError(
        'Available identifiers for stopword languages are: {}'.format(
            ', '.join(_STOPWORDS_SUPPORTED_LANGUAGES)
            ))
        else:
            stopwords = set(nltk_stopwords.words(stopwords_language))

        self._tokenizer = TweetTokenizer()
        self._regex_filters = regex_filters
        self._stopwords = stopwords
        self._filter_punctuation = filter_punctuation
        self._lower = lower

    def tokenize(self, text):
        """
        Tokenize text passed as argument
        """
        tokens = self._tokenizer.tokenize(text)

        if self._filter_punctuation:
            tokens = list(filter(lambda token: token not in string.punctuation, tokens))
        for regex_filter in self._regex_filters:
            tokens = list(filter(lambda token: not re.match(regex_filter, token), tokens))
        if self._lower:
            tokens = [token.lower() for token in tokens]
        if self._stopwords:
            tokens = list(filter(lambda token: token not in self._stopwords, tokens))

        return tokens

class Phraser(object):
    """
    Wrapper for Gensim Phrases: class to find common multiword
    expressions and join in in tokenized text
    """
    def __init__(self, tokenized_texts=None):
        """
        Initialize the object creating model to find phrases
        """
        self._model = Phrases(
            tokenized_texts,
            min_count=1,
            threshold=1,
            connector_words=ENGLISH_CONNECTOR_WORDS
            )

    def phrase(self, tokenized_text):
        """
        Takes tokenized text and returns the same list of
        tokens with phrases if found (a phrase is the join of to tokens)
        """
        tokenized_text_with_phrases = self._model[tokenized_text]

        return tokenized_text_with_phrases

class BinaryTextClassifierTrainer(object):
    """
    Define objects to train binary text classifiers (1/0)
    """

    def __init__(
        self, normalizer=None, tokenizer=None, default_hyperparameters=None
        ):
        """
        Initialize the object with specific utilities to
        preprocess text
        """
        self._normalizer = normalizer
        self._tokenizer = tokenizer

        self._check_default_hyperparameters(
            default_hyperparameters
        )
        self._default_hyperparameters = default_hyperparameters

    def _check_default_hyperparameters(self, hyperparameters):
        """
        Checks that all hyperparameters have been provided
        and in a proper way
        """

        for hyperparameter, type_ in _HYPERPARAMETERS_DESCRIPTION.items():
            if hyperparameter not in hyperparameters:
                raise AttributeError(
            'Mandatory hyperparameter {} not provided'.format(hyperparameter)
            )

        self._check_hyperparameters(hyperparameters)

    def _check_hyperparameters(self, hyperparameters):
        """
        Checks that hyperparameters provided are correct
        """
        for hyperparameter in hyperparameters:
            if hyperparameter not in _HYPERPARAMETERS_DESCRIPTION:
                raise AttributeError(
            'Unknown item {} provided as hyperparameter'.format(
                hyperparameter
                )
            )

            value = hyperparameters[hyperparameter]
            type_ = _HYPERPARAMETERS_DESCRIPTION[hyperparameter]

            if not isinstance(value, type_):
                raise ValueError(
            'Wrong value {} provided for hyperparameter {}'.format(
                str(value), hyperparameter
                )
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
            for hyperparameter, value in self._default_hyperparameters.items:
                if hyperparameter not in hyperparameters:
                    hyperparameters[hyperparameter] = value
        return hyperparameters

    def _create_model(self, hyperparameters=None):
        """
        Create a model to be trained
        """
        num_words = hyperparameters['num_words']
        n_hidden_layers = hyperparameters['n_hidden_layers']
        hidden_layer_size = hyperparameters['hidden_layer_size']
        dropout = hyperparameters['dropout']
        hidden_act_fun = 'relu'

        model = Sequential()

        # For simple logistic regresion without hidden layers
        if n_hidden_layers < 1:
            model.add(Dense(1, input_shape=(num_words,)))
        # For multilayer perceptron with 1 or more hidden layers
        else:
            # First hidden layer
            model.add(Dense(
                hidden_layer_size, input_shape=(num_words,),
                activation=hidden_act_fun
                ))
            model.add(Dropout(dropout))
            # More hidden layers if any
            for _ in range(n_hidden_layers - 1):
                model.add(Dense(
                    hidden_layer_size, activation=hidden_act_fun
                    ))
                model.add(Dropout(dropout))
            # Output layer
            model.add(Dense(1))

        model.add(Activation("sigmoid"))

        model.compile(
            loss="binary_crossentropy", optimizer="adam",
            metrics=["accuracy", Precision(), Recall()]
            )

        model.summary()

        return model

    def _save_model(self, model=None, vectorizer=None, phraser=None, output_folder=None):
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

    def _check_training_parameters(
        self, training_set=None, validation_set=None, hyperparameters=None,
        output_folder=None, text_field=None, label_field=None
        ):
        """
        Checks if input parameters for train method are correct
        """
        if not (text_field and label_field):
            raise AttributeError(
        'No mandatory parameters text_field or label_field provided'
        )

        for dataset in (training_set, validation_set):
            if dataset is None:
                raise AttributeError(
            'Both training_set and evaluation_set must be provided'
            )
            if not isinstance(dataset, pd.DataFrame):
                raise ValueError('Datasets must be DataFrame objects')
            if text_field not in dataset.columns:
                raise AttributeError(
            'No column {} provided in dataset'. format(text_field)
            )
            if label_field not in dataset.columns:
                raise AttributeError(
            'No column {} provided in dataset'. format(text_field)
            )

        if hyperparameters and not isinstance(hyperparameters, dict):
            raise ValueError('Hyperparameters must be a dictionary')

        if not (output_folder and isinstance(output_folder, str)):
            raise ValueError('Output folder must be a string')

    def train(
        self, training_set=None, validation_set=None, hyperparameters=None,
        output_folder=None, text_field=None, label_field=None
        ):
        """
        Trains a model given training data and a specific
        set of hyperparameters
        """

        self._check_training_parameters(
            training_set=training_set, validation_set=validation_set,
            hyperparameters=hyperparameters, output_folder=output_folder,
            text_field=text_field, label_field=label_field
        )

        if not os.path.isdir(output_folder):
            try:
                os.makedirs(output_folder)
            except OSError:
                raise OSError('Unable to create output folder {}')

        hyperparameters = self._get_hyperparameters(hyperparameters)

        num_words = hyperparameters['num_words'] 
        batch_size = hyperparameters['batch_size']
        epochs = hyperparameters['epochs']

        train_texts = training_set[text_field].to_list()
        train_texts = [self._normalizer.normalize(text) for text in train_texts]
        train_texts = [self._tokenizer.tokenize(text) for text in train_texts]
        val_texts = validation_set[text_field].to_list()
        val_texts = [self._normalizer.normalize(text) for text in val_texts]
        val_texts = [self._tokenizer.tokenize(text) for text in val_texts]
        train_labels = training_set[label_field].to_list()
        val_labels = validation_set[label_field].to_list()

        vectorizer = text.Tokenizer(num_words=num_words)
        vectorizer.fit_on_texts(train_texts)
        x_train = vectorizer.texts_to_matrix(train_texts, mode="tfidf")
        y_train = np.array(train_labels)
        x_val = vectorizer.texts_to_matrix(val_texts, mode="tfidf")
        y_val = np.array(val_labels)

        model = self._create_model(hyperparameters=hyperparameters)
        model.fit(
            x_train, y_train, validation_data=(x_val, y_val),
            batch_size=batch_size, epochs=epochs, verbose=2
            )
        self._save_model(model=model, vectorizer=vectorizer, output_folder=output_folder)
        metrics = self._evaluate_model(model=model, x=x_val, y=y_val)

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
        x = self._vectorizer.texts_to_matrix(texts, mode="tfidf")

        predictions = self._model.predict(x)
        if len(predictions) == 1:
            predictions = predictions[0]

        return predictions
