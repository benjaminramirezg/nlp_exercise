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

The dataset proposed in the exercise can be found in `data/Tweets.xlsx`. It was originaly provided in xlsx format. I exported it to raw csv format, a more handy format to be processed by code: `data/Tweets.csv`.

I've done an exploratory analysis of the dataset. The code used to do that exploratoy analysis can be found in the notebook `notebooks/dataset_analysis.csv`. It has been prepared to be self-explanatory.

The analysis followed these steps:

- Checking the size of the dataset.
- Checking the consistency of the data model in the dataset.
- Checking if there are many duplicates
- Checking if the dataset is balanced.
- Checking the nature and quality of the text.
- Checking the content of the texts.

The conclusions of the exploratory analysis have been the following:

- The dataset is highly unbalanced, but there may be enough instances of the positive class to undersample the negative one and train a model with a balanced version of the dataset. 
- Texts need certain cleaning and normalization. It has UTF-8 encoding but some Unicode characters and break lines appear escaped. There are also some characters that cannot be correctly encoded in UTF-8. 
- No duplicates found.
- There are texts in different languages, but the vast majority of them are in English, so, at least for a first baseline approach, not multilingual approach will be assumed.
- Positive texts seem to have very characteristic words and phrases, so a Bag of Words representation of text may be enough to fed a binary classifier, at least for a first baseline version.


## Solution

Given the conclusions of the exploratory analysis of the dataset, I have decided to follow the apprach that I explain in this section.

#### Unbalanced dataset

In order to tackle the problem of an unbalanced dataset, different strategies can be followed: undersampling the most populated class, oversampling the less populated, try a weighted loss function or try algorithms more robust with unbalanced datasets such as Random Forest, try to modelize the problem as an anomaly detection one...

The easiest solution is undersampling the most populated class. For that, you need to have enough instances of the less populated class. In this case, we have ~17000 instances of the positive class. A resulting balanced dataset would have ~33000 instances. It may be enough to train a good binary classifier, at least as a baseline version. So I've decided to undersample the negative class and use a balanced version of the dataset to train my models. In order to make sure that the resulting selection of instances of the negative class is not biased, I've randomly shuffled the instances before I select the first ~17000.  

#### Texts in different languages

In the dataset there are texts in different languages. However, the vast majority are in English (only ~4000 are in other languages). So I've decided not giving a special treatment to non-English texts in a first version.

In future versions, several improvements could be done. We could automaticaly translate non English texts (maybe by means of several [Huggingface](https://huggingface.co/models?pipeline_tag=translation) open models). In that case, a previous language identification should be done, maybe with the [FastText](https://fasttext.cc/docs/en/language-identification.html) language identifier. Other approach coud be using a multilingual encoder ([multilingual BERT](https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1), [Sentence Transformers](https://www.sbert.net), [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/1)) to represent text.

#### Preprocessing: cleaning and tokenization

In order to clean, normalize and tokenize the text I have written two classes in the `src/main/python/models.py` library:
- `TextNormalizer`: intended to:
    - Unescape escaped unicode characters
    - Remove break of lines
    - Remove non UTF-8 characters
- `TextTokenizer`: intended to:
    - Splits text into tokens where specific Twitter tokens such as hashtags and user names are preserved as tokens.
    - Filter unwanted tokens by means of regular expresions and a stopwords list.
    - Lower case tokens.

#### Vectorization of the texts as Bag of Words

The text must be vectorized in a way that captures characteristic features of the pro-ISIS texts. I have seen in the exploratory analysis of the dataset that pro-ISIS texts tend to have very specific words and phrases. So maybe a representation of texts as a Bag of Words is enough to capture the characteristic features of those texts. No context-sensitive approaches such as RNN or encoders based on _attention_ (transformers) seem to be needed, at least for a first version.

So the solution that I propose for the vectorization of texts is a bag of words where words are valued with _TF-IDF_ (Term Frequency-Inverse Document Frequency), that gives a sense of how relevant is every word of the BoW in every specific text.

The BoW are calculated for pre-tokenized texts by means of a Keras `Tokenizer` object.

#### Multilayer Percepton or Logistic Regression

Our models will define a function able to map a text into a float between 1 and 0 that will be used as the probability that the text is pro-ISIS.

Depending on the complexity of that relation, we will need to modelize it as a linear function or as a more complex one. I consider that we have enough data to train simple neural networks. So I decided to create in `src/main/python/models.py` a class `BinaryTextClassifierTrainer` that lets you train more or less complex Neural Networks according to a specific configuration file. Then, the idea is training different models with different configurations and decide what model works better on empirical basis: 

- I've trained a first model `models/modellr01` according to the configuration of `config/config.lr01.json`. Given that configuration, the model trained is, in fact, a simple logistic regression one, without hidden layers, with just an output neuron.

- I've also trained a model `models/modelmlp01` that has been configured according to the configuration of `config/config.mlp01.json`. Given that configuration it is a multilayer perceptron with a hidden layer of 250 neurons and the needed output layer with a single neuron.

- Anyway, the output layer is always a one-neuron with a _sigmoid_ activation function, as needed in binary classification problems. This function proyects the logits of the output layer neuron in a range between 1 and 0.   

- `BinaryTextClassifierTrainer` uses Tensorflow to create and train the models.

## Evaluation

I've tried different hyperparameter settings to train different models, but I have mainly compared two types of models: a logistic regression one a multilayer perceptron:

 - `models/modellr01`: a logistic regression model
 - `models/modelmlp01`: a multilayer perceptron

#### Metrics

I have used three metrics: accuracy, precision, recall and F1_score:

- Accuracy is the proportion of right results over the total amount of instances tested
- Precision id the proportion of right results over the amount of instances predicted as positive
- Recall is the proportion of right results over the amount of instances that should have been predicted as positives.
- F1_score is a combined measure of precision and recall that let you have a single scoring to describe how skilled is the model

Mainly the combination and recall gives us a precise evaluation of the performance of the model.

#### Validation during training

During the training process, Keras shows you the loss and metrics evaluated over in the training set and the validation set per every epoch of the training. For the logistic regression model, I got something similar to this:

```
Epoch 1/10
loss: 0.3083 - precision: 0.9295 - recall: 0.8694 - val_loss: 0.2049 - val_precision: 0.9696 - val_recall: 0.8913
Epoch 2/10
loss: 0.1534 - precision: 0.9810 - recall: 0.9298 - val_loss: 0.1684 - val_precision: 0.9660 - val_recall: 0.9143
Epoch 3/10
loss: 0.1135 - precision: 0.9845 - recall: 0.9467 - val_loss: 0.1551 - val_precision: 0.9680 - val_recall: 0.9192
Epoch 4/10
loss: 0.0920 - precision: 0.9876 - recall: 0.9549 - val_loss: 0.1510 - val_precision: 0.9668 - val_recall: 0.9210
Epoch 5/10
loss: 0.0778 - precision: 0.9904 - recall: 0.9603 - val_loss: 0.1515 - val_precision: 0.9638 - val_recall: 0.9216
Epoch 6/10
loss: 0.0681 - precision: 0.9913 - recall: 0.9634 - val_loss: 0.1537 - val_precision: 0.9632 - val_recall: 0.9210
Epoch 7/10
loss: 0.0607 - precision: 0.9923 - recall: 0.9659 - val_loss: 0.1585 - val_precision: 0.9626 - val_recall: 0.9228
Epoch 8/10
loss: 0.0550 - precision: 0.9928 - recall: 0.9688 - val_loss: 0.1631 - val_precision: 0.9609 - val_recall: 0.9265
Epoch 9/10
loss: 0.0505 - precision: 0.9938 - recall: 0.9697 - val_loss: 0.1701 - val_precision: 0.9590 - val_recall: 0.9241
Epoch 10/10
loss: 0.0468 - precision: 0.9935 - recall: 0.9717 - val_loss: 0.1779 - val_precision: 0.9589 - val_recall: 0.9204

```

After that, I tried a deeper model, a multilayer perceptron. I configured it with a single hidden layer with 250 neurons. The result is similar to this:

```
Epoch 1/10
loss: 0.2061 - precision: 0.9341 - recall: 0.9025 - val_loss: 0.1551 - val_precision: 0.9547 - val_recall: 0.9222
Epoch 2/10
loss: 0.0887 - precision: 0.9765 - recall: 0.9554 - val_loss: 0.1650 - val_precision: 0.9597 - val_recall: 0.9271 
Epoch 3/10
loss: 0.0513 - precision: 0.9889 - recall: 0.9710 - val_loss: 0.2057 - val_precision: 0.9530 - val_recall: 0.9247 
Epoch 4/10
loss: 0.0340 - precision: 0.9940 - recall: 0.9781 - val_loss: 0.2358 - val_precision: 0.9469 - val_recall: 0.9320 
Epoch 5/10
loss: 0.0258 - precision: 0.9954 - recall: 0.9830 - val_loss: 0.2941 - val_precision: 0.9517 - val_recall: 0.9210 
Epoch 6/10
loss: 0.0202 - precision: 0.9968 - recall: 0.9857 - val_loss: 0.3279 - val_precision: 0.9493 - val_recall: 0.9216 
Epoch 7/10
loss: 0.0174 - precision: 0.9967 - recall: 0.9873 - val_loss: 0.3828 - val_precision: 0.9503 - val_recall: 0.9186 
Epoch 8/10
loss: 0.0160 - precision: 0.9975 - recall: 0.9879 - val_loss: 0.4016 - val_precision: 0.9459 - val_recall: 0.9241 
Epoch 9/10
loss: 0.0144 - precision: 0.9978 - recall: 0.9887 - val_loss: 0.4201 - val_precision: 0.9466 - val_recall: 0.9253 
Epoch 10/10
loss: 0.0135 - precision: 0.9979 - recall: 0.9889 - val_loss: 0.4613 - val_precision: 0.9417 - val_recall: 0.9216 

```

Conclusions:

- Both models perform in a very similar way.
- Scores of precision tend to be higher that 95%, and scores of recall tend to be higher that 92%
- After the first 5 epochs the models overfit, because metrics in training set get closer to 1 but metrics in validation set doesn't improve, even decrease. So they learn specific too idiosincrasies features of the trainingset, and don't generalize.

Given those conclusions I've been adjusting the hyperparamaters configuring the corresponding configuration files. The final settings can be seen in `config`.   

#### Final evaluation over testing set

The final evaluation of the two models (linear regression and multilayer perceptron) has been done in the notebook `notebooks/evaluation_model.ipynb`. It has been prepared to be self-explanatory. In the notebook can be seen the following steps_

- Predicting with the two models over new text no used before:  `data/dataset/testing_set.csv`
- Calculating the optimal thresholds for every model.
- Given that threshold, I convert the scores of the prediction into binary labels (0/1)
- By comparing the original labels of the dataset with the predicted labels, precision, recall and F1 score are calculated for the two models.
- Displaying resoults with Precision-Recall curves

The results are as follows:

| Metrics   | Logistic Regression  |  Multilayer Perceptron |
|-----------|:--------------------:|-----------------------:|
| Precision |          0.964       |           0.963        |
| Recall    |          0.906       |           0.910        |
| F1 score  |          0.934       |           0.936        |


The conclusions about the performance of the model depend on the requirements of the specific use case in which the model would be used. In a specific use case, the recall score, for example may be not high enough, because finding almos all pro-ISIS tweets is critical. But in general, we can say the the performance seems relatively good.


## Next steps