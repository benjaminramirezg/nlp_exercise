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

The dataset proposed in the exercise can be found in `data/Tweets.xlsx`. It was originaly provided in _xlsx_ format. I was then saved also in raw csv format, a more handy format to be processed by code: `data/Tweets.csv`.

## Solution

## Evaluation

## Next steps