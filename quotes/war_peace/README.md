# Quotes classification on two themes
> Author: Gilles Schneider

## About
Train a classifier on a dataset of quotes. We focus on two different themes (war and peace here).

Dataset used: [Quotes-500K](https://github.com/ShivaliGoel/Quotes-500K)


Title: Proposing Contextually Relevant Quotes for Images

Authors: Shivali Goel, Rishi Madhok, Shweta Garg

In proceedings of: 40th European Conference on Information Retreival
Year: 2018

## Dependencies
Please make sure that you have all the dependencies installed.

```sh
pandas, numpy, random, sklearn, tqdm, nltk
```

## Installation
- Download Dataset [here](https://goo.gl/R3Sa34)
- Put **quotes_dataset.csv** in /war_peace
- Run datasets.py to generate the new Quotes-500 dataset

## Files
1. [dataset.py](/datasets.py): compute a new dataset with two types of quotes from **Quote-500K** (see class **DatasetTwoThemes** for more information)
2. [training.py](/training.py): train and test a classifier on the new dataset

## New Quotes-500
The new dataset is in a .csv file with three columns *quote*, *meta* and *themes*. 

- *quote*: quote
- *meta*: author and title
- *themes*: themes of the quote

## Preprocessing
The following preprocessing operations can be performed on quotes:

- Remove stop words
- Stemming

The influence of the preprocessing operations are studied in the section **Results (peace vs war)**. 

## Training Pipeline
The pipeline is described below:
1. *Apply preprocessing operations and balance the dataset by randomly selecting the same number of quotes.*
2. [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold): *Split the dataset into a train set and test set, preserve the percentage of samples for each class (folds are balanced), 10 folds have been used.*
3. [CountVectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html): *Training: convert a collection of quotes to a matrix of token counts.*
4. [TfidTransformer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html): *Training: transform a count matrix to a normalized tf or tf-idf representation.*
5. [MLPClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html): *Multi-layer Perceptron classifier.*


## Results (peace vs war)
A **Multi Layer Perceptron (MLP)** with 2 hidden layers of size 15 has been trained and tested on the new Quote-500k dataset. The training parameters remained the same accross the training/testing sessions. In the case of *war vs peace*, 623 peace quotes out of 2463 are randomly selected at each session. To cover as many peace quotes as possible, I ran 100 sessions for different configurations: same neural network, learning parameters, etc. but different preprocessing operations. 


**Balanced Dataset**
| Number of Sessions       | Preprocessing Operations           | Avg Accuracy  (Test)|
| ------------- |:-------------:| -----:|
| 100    | With stop words, no stemming |   0.8266 ± 0.0216
| 100    | With stop words, stemming |   0.8231 ± 0.01973
| 100    | Without stop words, no stemming |   0.8218 ± 0.0229
| 100    | Without stop words, stemming |   0.8209 ± 0.02186

---

## Meta
Gilles Schneider – [My website](https://gillesschneider.github.io/me/)



<!-- Markdown link & img dfn's -->
[nlp-image]: https://github.com/GillesSchneider/natural-language-processing/
[nlp-url]: https://github.com/GillesSchneider/natural-language-processing/
