from copy import Error
import pandas as pd
import json
import numpy as np
import os
from collections import Counter
import warnings
import re
import random

# Use DatasetTwoThemes in the following order
# dataset = DatasetWarPeace()
# dataset._clean()
# dataset._filter()

class DatasetTwoThemes(object):

    def __init__(self, path = "quotes_dataset.csv"):
        data = pd.read_csv("quotes_dataset.csv")
        # remove first row and useless columns
        data = data.iloc[:, 0:3].dropna()
        self._data = data # .json file
        self._len = 0 # len = number of quotes
        self._dataset = None # csv of quotes

    # extract war and peace quotes
    # save everything to a cleaned .csv
    def _clean(self, path = "dataset.csv", overwrite = True):

        # Clean the Quotes-500K
        # path: path to the new Quotes-500K dataset
        # overwrite: overwrite the file at 'path' if it exists, default: True

        self._data.columns = ['quote', 'meta', 'themes']
        self._data['quote'] = self._data['quote'].str.strip()
        self._data['meta'] = self._data['meta'].str.strip()
        self._data['themes'] = self._data['themes'].str.strip()

        # do some cleaning
        self._data['quote'] = self._data['quote'].replace(r",", r" ", regex=True)
        self._data['quote'] = self._data['quote'].replace(r'(?<=[.,?!;:])(?=[^\s])', r' ', regex=True)
        self._data['quote'] = self._data['quote'].replace(r"(?<=\w)([A-Z])", r" \1", regex=True)
        dataset = self._data

        # overwrite it if already exists and overwrite is true
        if os.path.exists(path):
            if overwrite:
                dataset.to_csv(path)
        else:
            dataset.to_csv(path)

        self._dataset = pd.read_csv(path, index_col=0).dropna()
        self._len = len(self._dataset)

    def _filter(self, path = "dataset.csv", themes = ['war', 'peace'], 
                show_stats = True, overwrite = True, reset_index = True):
        # Filter the dataset to keep only quotes with 'themes'
        # path: path to the dataset to filter, default: 'dataset.csv'
        # themes: list of length 2 (two themes), default: ['war', 'peace']
        # show_stats: print distribution of quotes, default: True
        # overwrite: overwrite the file that can be found at path with filtered dataset, default: True

        stats = {
            "number of quotes": 0,
            str(themes[0]) + " quotes": 0,
            str(themes[1]) + " quotes": 0,
        }

        if os.path.exists(path):
            
            # remove quotes that contain themes[0] and themes[1]
            filter = self._dataset[(self._dataset['themes'].str.contains(themes[0]))
                                            & (self._dataset['themes'].str.contains(themes[1]))]
            self._dataset.drop(filter.index, inplace = True)
            self._len = len(self._dataset)
            stats['number of quotes'] = self._len

            # keep quotes that contain themes[0] or themes[1]
            self._dataset = self._dataset[self._dataset['themes'].str.contains(" " + str(themes[0]) + ",| " + str(themes[1]) + ",")]

            # replace list of themes with unique value
            self._dataset.loc[self._dataset['themes'].str.contains(" " + str(themes[0]) + ","), 'themes'] = themes[0]
            self._dataset.loc[self._dataset['themes'].str.contains(" " + str(themes[1]) + ","), 'themes'] = themes[1]

            # update stats
            stats[str(themes[0]) + " quotes"] = len(self._dataset[self._dataset['themes'] == themes[0]])
            stats[str(themes[1]) + " quotes"] = len(self._dataset[self._dataset['themes'] == themes[1]])
            
            # reset indexing
            if reset_index:
                self._dataset.reset_index(inplace = True, drop = True)    

            # overwrite path file
            if overwrite:
                self._dataset.to_csv(path)
            
            # show stats
            if show_stats:
                stats = pd.Series(stats)
                print(stats)
            
        else:
            raise Error("Please generate cleaned dataset first using _clean_dataset()")





