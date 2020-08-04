"""Test skift."""

import os
import pickle

import pytest
import pandas as pd
import tempfile

from sklearn.exceptions import NotFittedError

from skift import FirstColFtClassifier


@pytest.mark.parametrize("quantize", [True, False])
def test_pickle(quantize):
    ftdf = pd.DataFrame(
        data=[['woof woof', 0], ['meow meow', 1]],
        columns=['txt', 'lbl']
    )
    ft_clf = FirstColFtClassifier()
    ft_clf.fit(ftdf[['txt']], ftdf['lbl'])
    if quantize:
        with pytest.raises(ValueError):
            ft_clf.quantize(cutoff=1)
        assert not ft_clf.is_quantized()
        return

    assert ft_clf.predict([['woof woof']])[0] == 0
    assert ft_clf.predict([['meow meow']])[0] == 1
    assert ft_clf.predict([['meow']])[0] == 1
    assert ft_clf.predict([['woof lol']])[0] == 0
    assert ft_clf.predict([['meow lolz']])[0] == 1

    fd, pic_fpath = tempfile.mkstemp()
    with open(pic_fpath, 'wb+') as bfile:
        pickle.dump(ft_clf, bfile)
    with open(pic_fpath, 'rb') as bfile:
        ft_clf2 = pickle.load(bfile)

    assert ft_clf2 != ft_clf
    assert ft_clf2.predict([['woof woof']])[0] == 0
    assert ft_clf2.predict([['meow meow']])[0] == 1
    assert ft_clf2.predict([['meow']])[0] == 1
    assert ft_clf2.predict([['woof lol']])[0] == 0
    assert ft_clf2.predict([['meow lolz']])[0] == 1

    if quantize:
        assert not ft_clf2.is_quantized()

    # Clean up
    os.close(fd)    # Prevent a file-handle leak
    os.unlink(pic_fpath)


def test_pickle_unfitted():
    ftdf = pd.DataFrame(
        data=[['woof woof', 0], ['meow meow', 1]],
        columns=['txt', 'lbl']
    )
    ft_clf = FirstColFtClassifier()

    fd, pic_fpath = tempfile.mkstemp()

    with open(pic_fpath, 'wb+') as bfile:
        pickle.dump(ft_clf, bfile)
    with open(pic_fpath, 'rb') as bfile:
        ft_clf2 = pickle.load(bfile)

    with pytest.raises(NotFittedError):
        assert ft_clf.predict([['woof woof']])[0] == 0

    ft_clf.fit(ftdf[['txt']], ftdf['lbl'])
    assert ft_clf.predict([['woof woof']])[0] == 0
    assert ft_clf.predict([['meow meow']])[0] == 1
    assert ft_clf.predict([['meow']])[0] == 1
    assert ft_clf.predict([['woof lol']])[0] == 0
    assert ft_clf.predict([['meow lolz']])[0] == 1

    assert ft_clf2 != ft_clf
    with pytest.raises(NotFittedError):
        assert ft_clf2.predict([['woof woof']])[0] == 0

    ft_clf2.fit(ftdf[['txt']], ftdf['lbl'])
    assert ft_clf2.predict([['woof woof']])[0] == 0
    assert ft_clf2.predict([['meow meow']])[0] == 1
    assert ft_clf2.predict([['meow']])[0] == 1
    assert ft_clf2.predict([['woof lol']])[0] == 0
    assert ft_clf2.predict([['meow lolz']])[0] == 1

    # Clean up
    os.close(fd)    # Prevent a file-handle leak
    os.unlink(pic_fpath)
