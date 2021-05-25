import numpy as np
import random
import tqdm
import json
from datasets import DatasetTwoThemes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd

# parameters
sessions_nb = 1
themes = ['peace', 'war']
# load new Quote-500
Q500 = DatasetTwoThemes()
Q500._clean()
Q500._filter(themes=themes)
dataset = Q500._dataset

scores = {themes=themes}

# training and testing 
for i in tqdm.tqdm(range(sessions_nb)):
    # get quotes
    X_war = dataset[dataset['themes'] == themes[0]]['quote'].values.tolist()
    X_peace = dataset[dataset['themes'] == themes[1]]['quote'].values.tolist()

    # balance quotes
    min_nb = np.min([len(X_war), len(X_peace)])
    X_war = random.sample(X_war, min_nb)
    X_peace = random.sample(X_peace, min_nb)

    # generate train and test data
    X = X_war + X_peace
    y = ['war']*min_nb + ['peace']*min_nb
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # training
    text_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('mlp', MLPClassifier(
                            random_state=1,
                            learning_rate_init=0.001,
                            hidden_layer_sizes= (3, 15),
                            max_iter=300,
                            batch_size=64))
                        ])

    text_clf = text_clf.fit(X_train, y_train)

    # testing
    predicted2 = text_clf.predict(X_test)
    scores['session' + str(i)] = np.around(np.mean(predicted2 == y_test), 2)

# save results
with open('scores.json', 'w') as fp:
    json.dump(scores, fp)



