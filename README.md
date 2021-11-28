# nlp_exercise

This project is a proposal of solution for the exercise proposed in `NLP_exercise.pdf`. In short, the exercise provides a dataset of tweets labeled as 1 or 0, where 1 indicates that the tweet seems to be a pro-ISIS one and the 0 indicates that it doesn't. The goal is to build a solution able to take a tweet and retrieve the probability that it is a pro-ISIS one.

## Preparing the environment
For the solution of this exercise I have used a python evironment that can be recreated, for example, as a conda environment, by following the next steps:

#### Creating a conda environment _nlp_exercise_

Assuming that we are in the root of the project, the definition of the environment needed to run the different scripts and notebooks can be found in the `conda.yaml` file. It contains the version of python used, and its main dependencies. The virtual environment can be recreated by running the script `scripts/create_conda_env.sh`.

```
$ cd scripts
$ ./create_conda_env.sh
```

The name of the resulting environment will be _nlp_exercise_. Once it has been created, other two additional dependencies must be installed on it. To install them, the environment must be activated.

```
$ conda activate nlp_exercise
```

The first additional dependency is _FastText_. It will be used to perform language identification. It can be installed by running the script `scripts/install_fasttext.sh`.

```
$ ./install_fasttext.sh
```

The last dependency to be installed is _NLTK_ stopwords. It can be installed by running the script `scripts/install_stopwords.sh`

```
$ python install_stopwords.py
```

That's all. With all those dependencies installed, and the environment activated, the different scripts and notebooks in the exercise can be run.

## Quickstart

Assuming that we are in the root of the project, in the `it` folder there are several scripts that can be run to easily test the code.

#### Prepare the dataset

The script `it/test.01.sh` can be run to prepare the dataset that will be used to train and evaluate models.

```
$ cd it
$ ./test.01.sh
```

It takes as input the original dataset in csv format (`data/Tweets.csv`) and undersamples the negative class to get a balanced dataset that saves in (`data/BalancedTweets.csv`).

#### Training the model

The script `it/test.02.sh` can be run to train a model able to take a tweet and return the probability that it is pro-ISIS.

```
$ ./test.02.sh
```

It takes as input the `data/BalancedTweets.csv` dataset previously created and a config file that can be found in `config/config.01.json` and trains and evaluate the model. It will print the evaluation of the model in STDOUT and will save the resulting artifacts in the `models/model01` folder.

#### Predicting over new text

The script `it/test.03.sh` uses the previously created model `models/model01` to predict over the text of a dataset.

```
$ ./test.03.sh
```

It loads the model saved in `models/model01` and predicts for every texts in `data/BalancedTweets.csv` the probability that it is pro-ISIS. Saves a new version of the dataset with those probabilities in `data/Predictions.csv`

## Analysis

The dataset proposed in the exercise can be found in `data/Tweets.xlsx`. It was originaly provided in _xlsx_ format. It was then saved also in raw csv format, a more handy format to be processed by code: `data/Tweets.csv`.

The code used to do an exploratoy analysis of the dataset can be found in `notebooks/dataset_analysis.csv`. According the conclusions of the analysis, one specific approach will be used to solve the exercise.

The analysis followed the next steps:

- Checking the size of the dataset
- Checking the consistency of the data model in the dataset
- Checking if the dataset is balanced
- Checking the nature and quality of the text
- Checking the content of the texts

The conclusions of the exploratory analysis have been the following:

- The dataset is highly unbalanced, but there may be enough instances of the positive class to undersample the negative one and train a model with a balanced version of the dataset
- Texts need certain cleaning and normalization
- There are texts in different languages, but the vast majority of them have been written in English, so, at least for a first baseline approach not multilingual approach need to be assumed
- Positive texts seem to have very characteristic words and phrases, so a Bag of Words representation of text may be enough to fed a binary classifier


## Solution

Given the conclusions in the exploratory analysis of the dataset, one specific approach has been followed to solve the proposed exercise.

#### Unbalanced dataset
- Enough data to train a model with an undersampled balanced dataset
- Script `scripts/undersample.csv`

#### Texts in different languages
- For a baseline solution, given that the vast majority of texts is in English, no specific solution has been proposed.
- Further solutions with: more data, machine translation or multilingual encoders such as multilingual BERT, USE o SBERT

#### Preprocessing: cleaning and tokenization

- Several classes for cleaning and tokenize text have been implemented in the library `src/main/python/models.py`
- `TextNormalizer`: unescape escaped unicode characters, remove break of lines and non UTF-8 characters.
- `TextTokenizer`: wrapper for the NLTK `TeetsTokenizer` that splits text into tokens where specific Twitter tokens as hashtags and user names are preserved as tokens. It has been improved with the ability to filter unwanted tokens by means of regular expresions, a list of stopwords, and the ability to normalize tokens passing them to lower case. It is very important to reduce de size of the dictionary and avoid frequent but non relevant tokens and tring that the bag of words will contain the most relevant words to classify the texts.
- `TextPhraser`: wraper to a Gensim utility to find common multiword expressions.

#### Vectorization of the texts as a Bag of Words

- Given that we have enough data to train a model with a balanced dataset, we will modelate the problem as a binary classification one. We will build a model as a function that given a text returns the probability that that text is pro-ISIS.

- The text must be vectorized in a way that captures characteristic features of the pro-ISIS ones. As we have seen that pro-ISIS texts tend to have very specific words and phrases, a representation of text as the set of words that contain may be enough. No context-sensitive approach such as RNN or encoders based on _attention_ (transformers) will be needed.
- So the tokenized text is vectorized as a bag of words by means of a Keras `Tokenizer` object. Words in the bag of words will be representend by _TF-IDF_ (Term frequency-Inverse Document Frequency) that gives a sense of how relevant is the word in that text (the most the word appear in the text and the less appear in the whole dataset, the most is its relevance).
- The size of the bag of words can be configured. We have used 10000, because we have observed that the 10000 most common words are the most relevant

#### Multilayer Percepton with Sigmoid function

- Our model needs to define a function that is able to map a text into a float between 1 and 0 that represents the probability that the text is pro-ISIS.
- If we presume that that function can be linear to properly modelize the nature of the problem, we could propose a linear model such as a logistic regression model. Instead, if the relation may be a more complex one, we may use a multilayer perceptron to modelize it. I have decided to take the decision on empirical bases. For that, I've implemented a class `BinaryTextClassifierTrainer` intended to train binary classifiers of texts. It uses Tensorflow and lets configure the topology of the model and its training with several hyperparameters. You can configure a simple logistic regression model without hidden layers that modelize the problem as a linear function. Instead, you can also modelize the problem as a non-linear one with a multilayer perceptron.
- Anyway, te output layer is always a one-neuron one whese activation function is _sigmoid_, as usually in binary classification problems. This function proyects the logits of the output layer neuron in a range between 1 and 0.   

## Evaluation

- We have tried different hyperparameter settings to train the models, but we have mainly compared two types of models: a logistic regression one, and a multilayer perceptron.

- We have used three metrics: accuracy, precision and recall.
- Accuracy is the proportion of right results over the total amount of instances tested
- Precision id the proportion of right results over the amount of instances predicted as positive
- Recall is the proportion of right results over the amount of instances that should have been predicted as positives.

Mainly precision and recall give us a precise metric of the performance of the model


## Next steps