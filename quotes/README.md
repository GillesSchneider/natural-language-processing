# Quotes classification with two themes
> Author: Gilles Schneider

## About
Train a classifier on a dataset of quotes. We only on focus two different themes (war and peace here).

Dataset used: [Quotes-500K](https://github.com/ShivaliGoel/Quotes-500K)


Title: Proposing Contextually Relevant Quotes for Images

Authors: Shivali Goel, Rishi Madhok, Shweta Garg

In proceedings of: 40th European Conference on Information Retreival
Year: 2018


## Installation
- Download Dataset [here](https://goo.gl/R3Sa34)
- Put **quotes_dataset.csv** in /war_peace
- Run **datasets.py** to generate the new Quotes-500 dataset


## New Quotes-500
The new dataset is in a .csv file with three columns *quote*, *meta* and *themes*. 

- *quote*: quote
- *meta*: author and title
- *themes*: themes of the quote

## Files
1. [dataset.py](/datasets.py): compute a dataset with two quotes from **Quote-500K**
2. [training.py](/training.py): train the classifier, 10 sessions

## Pipeline
The pipeline is described below.

1. [CountVectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html): *Convert a collection of quotes to a matrix of token counts*
2. [TfidTransformer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html): *Transform a count matrix to a normalized tf or tf-idf representation*
3. [MLPClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html): *Multi-layer Perceptron classifier.*

## Dependencies

Please make sure that you have all the dependencies before using the code.

```sh
pandas, numpy, random, sklearn, tqdm
```

## Results (peace vs war)
The Multi Layer Perceptron (MLP) has 3 hidden layers of size 50, and has been trained on 300 iterations. 10 training and testing sessions. **Please note that quotes are randomly selected at each session.**

| Session       | Parameters           | Accuracy  (Test)|
| ------------- |:-------------:| -----:|
| 0    | LR: 0.01, Iter: 300 |   0.84
| 1    |  |   0.80
| 2    |  |   0.80
| 3    |  |   0.80
| 4    |  |   0.82
| 5    |  |   0.82
| 6    |  |   0.85
| 7    |  |   0.80
| 8    |  |   0.81
| 9    |  |   0.86




## Meta

Gilles Schneider – [My website](https://gillesschneider.github.io/me/)



<!-- Markdown link & img dfn's -->
[nlp-image]: https://github.com/GillesSchneider/natural-language-processing/
[nlp-url]: https://github.com/GillesSchneider/natural-language-processing/
