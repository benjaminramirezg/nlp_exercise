# ##############################################################################
#
# Version: 0.0.1 (24 November 2021)
# Author: Benjamín Ramírez (benjaminramirezg@gmail.com)
#
# ##############################################################################

"""
Library with classes to train text classification models and predict with them
"""

import json
import pickle
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import text

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dense, Activation, Dropout

_HYPERPARAMETERS_DESCRIPTION = {
    'num_words': int,
    'hidden_layer_size': int,
    'dropout': float,
    'train_percent': float,
    'batch_size': int,
    'epochs': int
}

class TextNormalizer(object):
    """
    Define objects intended to normalize text
    """
    def __init__(self, parameters=None):
        """
        Initialize the object with specific parameters
        """
        self._parameters = parameters

    def normalize(self, text):
        """
        Normalize text passed as argument
        """
        return text

class TextTokenizer(object):
    """
    Define objects intended to tokenize text
    """
    def __init__(self, parameters=None):
        """
        Initialize the object with specific parameters
        """
        self._parameters = parameters

    def tokenize(self, text):
        """
        Tokenize text passed as argument
        """
        tokens = []
        return tokens

class BinaryClassifierTrainer(object):
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
        except:
            raise Exception(OSError(
        'Unable to open config file {}'.format(filepath)
        ))

        hyperparameters = None
        try:
            hyperparameters = json.loads(file_content)
        except:
            raise Exception(ValueError(
        'Unable to parse json config file {}'.format(filepath)
        ))

        for hyperparameter, type_ in HYPERPARAMETERS_DESCRIPTION.items():
            if hyperparameter not in hyperparameters:
                raise Exception(AttributeError(
            'Mandatory hyperparameter {} not provided'.format(hyperparameter)
            ))
            value = hyperparameters[hyperparameter]
            if not isinstance(value, type_):
                raise Exception(ValueError(
            'Wrong value {} provided for hyperparameter {}'.format(str(value), hyperparameter)
            ))

        return hyperparameters

    def _get_hyperparameters(self, hyperparameters):
        """
        Return hyperparameter values if provided. Otherwise default
        values are provided
        """
        if hyperparameters is None:
            hyperparameters = self._default_hyperparameters
        else:
            for hyperparameter, type_ in HYPERPARAMETERS_DESCRIPTION.items():
                if hyperparameter not in hyperparameters:
                    hyperparameters[hyperparameter] = self._default_hyperparameters[hyperparameter]
                else:
                    value = hyperparameters[hyperparameter]
                    if not isinstance(value, type_):
                        raise Exception(ValueError(
                    'Wrong value {} provided for hyperparameter {}'.format(str(value), hyperparameter)
                    ))
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

        model_path = '{}/model.h5'.format(output_folder)
        vectorizer_path = '{}/vectorizer.pkl'.format(output_folder)
        normalizer_path = '{}/normalizer.pkl'.format(output_folder)
        tokenizer_path = '{}/tokenizer.pkl'.format(output_folder)

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
        hyperparameters = self._get_hyperparameters(hyperparameters)

        num_words = hyperparameters['num_words'] if 'num_words' in 
        batch_size = hyperparameters['batch_size']
        epochs = hyperparameters['epochs']
        train_percent = hyperparameters['train_percent']

        train_dataset, eval_dataset = self._split_train_test_texts(
            dataset=dataset, train_percent=train_percent
        )

        train_texts = train_dataset['text'].to_list()
        train_texts = [self._normalizer.normalizer(text) for text in train_texts]
        train_texts = [self._tokenizer.tokenizer(text) for text in train_texts]
        eval_texts = eval_dataset['text'].to_list()
        eval_texts = [self._normalizer.normalizer(text) for text in eval_texts]
        eval_texts = [self._tokenizer.tokenizer(text) for text in eval_texts]
        train_labels = train_dataset['label'].to_list()
        eval_labels = eval_dataset['label'].to_list()

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