"""Test skift."""

import os
import pickle

import pandas as pd
from skift import FirstColFtClassifier


def test_base_case():
    ftdf = pd.DataFrame(
        data=[['woof woof', 0], ['meow meow', 1]],
        columns=['txt', 'lbl']
    )
    ft_clf = FirstColFtClassifier()
    ft_clf.fit(ftdf[['txt']], ftdf['lbl'])

    assert ft_clf.predict([['woof woof']]) == 0
    assert ft_clf.predict([['meow meow']]) == 1
    assert ft_clf.predict([['meow']]) == 1
    assert ft_clf.predict([['woof lol']]) == 0
    assert ft_clf.predict([['meow lolz']]) == 1

    pic_fpath = os.path.expanduser('~/.temp/ttemp_ft_model.ft')
    with open(pic_fpath, 'wb') as bfile:
        pickle.dump(ft_clf, bfile)
    with open(pic_fpath, 'rb') as bfile:
        ft_clf2 = pickle.load(bfile)

    assert ft_clf2 != ft_clf
    assert ft_clf2.predict([['woof woof']]) == 0
    assert ft_clf2.predict([['meow meow']]) == 1
    assert ft_clf2.predict([['meow']]) == 1
    assert ft_clf2.predict([['woof lol']]) == 0
    assert ft_clf2.predict([['meow lolz']]) == 1
