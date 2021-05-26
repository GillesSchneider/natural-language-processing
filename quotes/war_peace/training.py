import numpy as np
import random
import tqdm
import json
from datasets import DatasetTwoThemes
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd

# parameters
sessions_nb = 100
themes = ['peace', 'war']

# load new Quote-500
Q500 = DatasetTwoThemes()
Q500._clean()
Q500._filter(themes=themes)
Q500._preprocess(stop_words=False, stemming=False)
dataset = Q500._dataset



# training and testing 
for i in tqdm.tqdm(range(sessions_nb)):
    X_war = dataset[dataset['themes'] == themes[0]]['quote'].values.tolist()
    X_peace = dataset[dataset['themes'] == themes[1]]['quote'].values.tolist()

    # get min number of quotes
    min_nb_quotes = np.min([len(X_peace), len(X_war)])

    X_war = random.sample(X_war, min_nb_quotes)
    X_peace = random.sample(X_peace, min_nb_quotes)

    # estimator
    text_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('mlp', MLPClassifier(
                            random_state=1,
                            learning_rate_init=0.001,
                            hidden_layer_sizes= (4, 15),
                            max_iter=300,
                            batch_size=32))
                        ])

    X = X_war + X_peace
    y = ['war']*len(X_war) + ['peace']*len(X_peace)

    sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(text_clf, X, y, cv=sss)

    print("session " + str(i), "mean:", scores.mean(), "std:", scores.std())

# save results
with open('scores.json', 'w') as fp:
    json.dump(scores, fp)

print("avg:", np.mean(list(scores.values())))

