# Quotes classification (war vs. peace)
> Author: Gilles Schneider

## About
Train a classifier on dataset of quotes. We only focus on war and peace quotes.

Dataset used: [Quotes-500K](https://github.com/ShivaliGoel/Quotes-500K)


Title: Proposing Contextually Relevant Quotes for Images

Authors: Shivali Goel, Rishi Madhok, Shweta Garg

In proceedings of: 40th European Conference on Information Retreival
Year: 2018


## Installation
- Download Dataset [here](https://goo.gl/R3Sa34)
- Put **quote_dataset.csv** in [war_peace/datasets](/datasets)
- Run **dataset_cleaning.py**

## Files
1. dataset_cleaning.py: add columns titles, add white space after *.,?!;:* and split word of the form *myName* into *my Name*.
2. training.py: train classifier on war and peace quotes (MultiLayer Perceptron)

## Pipeline
The pipeline is described below.

1. [CountVectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html): *Convert a collection of text documents to a matrix of token counts*
2. [TfidTransformer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html): *Transform a count matrix to a normalized tf or tf-idf representation*
3. [MLPClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html): *Multi-layer Perceptron classifier.*

## Dependencies

Please make sure that you have all the dependencies before using the code.

```sh
pandas, numpy, random, sklearn
```


## Results
The Multi Layer Perceptron (MLP) has 3 hidden layer of size 50, and has been trained on 300 iterations. The accuracy is the mean accuracy computed on ten training/testing sessions. 

| Model        | Parameters           | Accuracy  (Test)|
| ------------- |:-------------:| -----:|
| MLP     | LR: 0.01, Iter: 300 | 0.79 |


## Meta

Gilles Schneider – [My website](https://gillesschneider.github.io/me/)



<!-- Markdown link & img dfn's -->
[nlp-image]: https://github.com/GillesSchneider/natural-language-processing/
[nlp-url]: https://github.com/GillesSchneider/natural-language-processing/
