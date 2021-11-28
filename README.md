# nlp_exercise

This project is a proposal of solution for the exercise proposed in `NLP_exercise.pdf`. In short, the exercise provides a dataset of tweets labeled as 1 or 0, where 1 indicates that the tweet seems to be a pro-ISIS one and the 0 indicates that it doesn't. The goal is to build a solution able to take a tweet and retrieve the probability that it is a pro-ISIS one.

## Preparing the environment
For the solution of this exercise I have used a python evironment that can be recreated as a conda environment by following the next steps:

Assuming that we are in the root of the project, the definition of the environment needed to run the different scripts and notebooks can be found in the `conda.yaml` file. It contains the version of python used, and its main dependencies. The virtual environment can be recreated by running the script `scripts/create_conda_env.sh`.

```
$ cd scripts
$ ./create_conda_env.sh
```

The name of the resulting environment will be _nlp_exercise_. Once it has been created, other two additional dependencies must be installed on it. To install them, the environment must be activated.

```
$ conda activate nlp_exercise
```

The first additional dependency is _FastText_. It will be used to do language identification. It can be installed by running the script `scripts/install_fasttext.sh`.

```
$ ./install_fasttext.sh
```

The last dependency to be installed is _NLTK_ stopwords. It can be installed by running the script `scripts/install_stopwords.sh`

```
$ python install_stopwords.py
```

With all  those dependencies installed, and the environment activated, the different scripts and notebooks in the exercise can be run.

## Structure of the project

```
nlp_exercise
│   README.md
│   conda.yaml    
│
└───scripts
│       ...
│
└───config
│       config.lr01.json
│       config.mlp01.json
│   
└───data
│    │   Tweets.xlsx
│    │   Tweets.csv
│    │
│    └────dataset
│             balanced_dataset.csv
│             training_set.csv
│             validation_set.csv
│             testing_set.csv
│             ...
│
└───it
│       test.01.sh
│       test.02.sh
│       test.03.sh
│       test.04.sh
│       test.05.sh
│
└───models
│       modellr01
│       modelmlp01
│ 
└───notebooks
│       dataset_analysis.ipynb
│       model_evaluation.ipynb
│
└───src/main/python
        models.py
        train.py
        predict.py
        prepare_dataset.py

```

- `src/main/python` contains the main code of the project. With this code you can prepare a dataset from the original `data/Tweets.csv` file, use it to train a model, and use that model to predict over new text: 
    - `prepare_dataset.py` is a script that takes an original dataset, undersamples it if needed to make sure that we have a balanced dataset, and splits it into training set, validation set and testing set.
    - `train.py` is a script to launch the training of a new model, given a dataset and a config file.
    - `predict.py` is a script that takes a previously trained model and a dataset, and predicts over the dataset according to the model.
    - `model.py` is the library in which the majority of the code has been written. The rest of scripts and notebooks just use the classes defined here. It contains, among others:
        - A class `TextNormalizer` intended to clean and normalize raw text (manage unicode characters and break lines for example)
        - A class `TextTokenizer` intended to tokenize raw text and filter non interesting tokens
        - A class `BinaryTextClassifierTrainer` intended to train models for binary classification of text
        - A class `BinaryTextClassifier` intended to load a previously trained model and predict over text
- `data` contains all the datasets used and created in the project:
    - `Tweets.xlsx` is the original dataset proposed in the exercise.
    - `Tweets.csv` is the same dataset exported in csv format, a more handy format to be used by code.
    - `datasets/` is a folder that contains a version of `Tweets.csv` suitable to be used to train and evaluate models. The original dataset was undersampled to make sure that it is a balanced one, and it was splitted into training, validation and testing sets. It is the result of executing `prepare_dataset.py` over `Tweets.csv`.
- `it` contains a bunch of integration tests. They may be specially interesting because contain examples of execution of the code in `src/main/python`. Specifically, they contain the commands that I used to create the dataset in `data/dataset` and the trained models in `models`.

- `config` contains the configuration settings that were used to train  different models. Specifically, `config/config.lr01.json` was used to train the model `models/modellr01` and `config/config.mlp01.json` was used to train the model `models/modelmlp01`.
- `model` is the folder where trained models have been saved
- `notebooks` contains two notebooks with the parts of the exercise that requiere visualization of plots:
    - In `dataset_analysis.ipynb` I show the preliminar exploratory analysis of the dataset that I did.
    - In `evaluation_model.ipynb` I show the evaluation that I did of the models `models/modellr01` and `models/modelmlp01` that I have trained.
- `scripts` contains several scripts that can be run to create a conda virtual environment with all dependencies needed to run the different scripts and notebooks in the project.

## Execution of the scripts

In this project I have trained two different models that can be used to take a tweet and predict if it is a pro-ISIS one or it isn't. The models have been saved in:

 - `models/modellr01`
 - `models/modelmlp01`

To train the models I have taken the original dataset `data/Tweets.csv` and I have preprocessed in a specific way. The result of that preprocessing was saved in the folder `data/dataset`.

I have also used the trained models to predict over new text and evaluate the models.

#### Preparation of the dataset

The preparation of the dataset has been done by executing the following commands (we assume that we are in the root of the project)

```
$ cd src/main/python
$ python prepare_dataset.py -d ../../../data/Tweets.csv -s ',' -t 'Tweet' -l 'ISIS Flag' -p 0.80 -o ../../../data/dataset/
```

The idea is that we take as input the original dataset `data/Tweets.csv` (parameter `-d`), the delimiter of fields of that csv file is `,` (parameter `-s`), the field of the csv where texts must be found is `Tweet` (parameter `-t`), the field where labels must be found is `ISIS Flag` (parameter `-l`), the proportion of data to be used in the training set is `0.80` (parameter `-p`), and the folder where resulting datasets will be saved is `data/dataset/`.   

The result is a set of files saved in `data/dataset/` with a balanced version of the original dataset and the splits of that balanced dataset in training, validation and testing sets (with a proportion of data of 80%, 10% and 10% respectively)   

Notice that we have an example of this command in `it/test.01.sh`.

#### Training of the models

I have trained two different models in order to see what type pf model performs better.

 - `models/modellr01`: a logistic regression model
 - `models/modelmlp01`: a multilayer perceptron

Assuming that we are in the root of the project, to train the `models/modellr01` model I have run the follwing commnds: 

```
$ cd ../src/main/python/.
$ python train.py -d ../../../data/dataset/training_set.csv -v ../../../data/dataset/validation_set.csv -s ',' -t 'Tweet' -l 'ISIS Flag' -o ../../../models/modellr01 -c ../../../config/config.lr01.json
```

The idea is to train a model according to the configuration that was saved in the file `config/config.01lr` (parameter `-c`). That file contains the configuration needed to train just a logistic regression model. The training data to be used is `data/dataset/training_set.csv` (parameter `-d`), and the validation set is `data/dataset/validation_set.csv` (parameter `-v`). The resulting model will be saved in `models/modellr01`.

The second model that has been trained is a multilayer perceptron. It has been trained and evaluated with the same datasets, but it used a different configuration (parameter `-c`) and was saved in a diferent folder (parameter `-o`)

```
$ cd ../src/main/python/.
$ python train.py -d ../../../data/dataset/training_set.csv -v ../../../data/dataset/validation_set.csv -s ',' -t 'Tweet' -l 'ISIS Flag' -o ../../../models/modelmlp01 -c ../../../config/config.mlp01.json
```

Notice that, scripts `it/test.02.sh` and `it/test.03.sh` contain those commands.

#### Predicting over new text

Once the models have been trained, they can be used to predict over new text. Specifically, the testing set `data/dataset/testing_set.csv` previously created can be used to to test our models.

With the following commands the model `models/modellr01` can be loaded and used to predict over that dataset:

```
cd ../src/main/python/.
python predict.py -d ../../../data/dataset/testing_set.csv -s ',' -t Tweet -l Predictions -m ../../../models/model_lr01 -o ../../../data/dataset/predictions_lr01.csv
```

The output of the process is a new dataset `data/dataset/predictions_lr01.csv` whith the texts and labels of the input dataset plus a new column `Predictions` with the scores predicted by the model.

The corresponding execution with the model `models/modelmlp01` is as follows:

```
cd ../src/main/python/.
python predict.py -d ../../../data/dataset/testing_set.csv -s ',' -t Tweet -l Predictions -m ../../../models/model_lr01 -o ../../../data/dataset/predictions_lr01.csv
```

Those execution can be found also in the scripts `it/test.04.sh` and `it/test.05.sh`.

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