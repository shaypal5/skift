
"""scikit-learn classifier wrapper for fasttext."""

import os
import abc

import numpy as np
from fasttext import train_supervised
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError

from .util import (
    temp_dataset_fpath,
    dump_xy_to_fasttext_format,
    python_fasttext_model_to_bytes,
    bytes_to_python_fasttext_model,
)


class FtClassifierABC(BaseEstimator, ClassifierMixin, metaclass=abc.ABCMeta):
    """An abstact base class for sklearn classifier adapters for fasttext.

    Parameters
    ----------
    **kwargs
        Keyword arguments will be redirected to fasttext.train_supervised.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kwargs.pop('input', None)  # remove the 'input' arg, if given
        self.model = None

    def __getstate__(self):
        if self.model is not None:
            model_pickle = python_fasttext_model_to_bytes(self.model)
            pickle_dict = self.__dict__.copy()
            pickle_dict['model'] = model_pickle
            return pickle_dict
        return self.__dict__

    def __setstate__(self, dicti):
        for key in dicti:
            if key == 'model':
                unpic_model = bytes_to_python_fasttext_model(dicti[key])
                setattr(self, 'model', unpic_model)
            else:
                setattr(self, key, dicti[key])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        # re-implementation that will preserve ft kwargs
        return self.kwargs

    ALLOWED_DTYPES_ = ['<U26', object]

    @staticmethod
    def _validate_x(X):
        try:
            if len(X.shape) != 2:
                raise ValueError(
                    "FastTextClassifier methods must get a two-dimensional "
                    "numpy array (or castable) as the X parameter.")
            return X
        except AttributeError:
            return FtClassifierABC._validate_x(np.array(X))

    @staticmethod
    def _validate_y(y):
        try:
            if len(y.shape) != 1:
                raise ValueError(
                    "FastTextClassifier methods must get a one-dimensional "
                    "numpy array as the y parameter.")
            return np.array(y)
        except AttributeError:
            return FtClassifierABC._validate_y(np.array(y))

    @abc.abstractmethod
    def _input_col(self, X):
        pass  # pragma: no cover

    def fit(self, X, y):
        """Fits the classifier

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        self._validate_x(X)
        y = self._validate_y(y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.num_classes_ = len(self.classes_)
        self.class_labels_ = [
            '__label__{}'.format(lbl) for lbl in self.classes_]
        # Dump training set to a fasttext-compatible file
        temp_trainset_fpath = temp_dataset_fpath()
        input_col = self._input_col(X)
        dump_xy_to_fasttext_format(input_col, y, temp_trainset_fpath)
        # train
        self.model = train_supervised(
            input=temp_trainset_fpath, **self.kwargs)
        # Return the classifier
        try:
            os.remove(temp_trainset_fpath)
        except FileNotFoundError:  # pragma: no cover
            pass
        return self

    @staticmethod
    def _clean_label(ft_label):
        return int(ft_label[9:])

    def _predict(self, X, k=1):
        # Ensure that fit had been called
        if self.model is None:
            raise NotFittedError("This {} instance is not fitted yet.".format(
                self.__class__.__name__))

        # Input validation{
        self._validate_x(X)
        input_col = self._input_col(X)

        return (self.model.predict(text, k) for text in input_col)

    def predict(self, X):
        """Predict labels.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            Predicted labels for the given inpurt samples.
        """
        return np.array([
            self._clean_label(res[0][0])
            for res in self._predict(X)
        ], dtype=np.float_)

    def _format_probas(self, result):
        lbl_prob_pairs = zip(result[0], result[1])
        sorted_lbl_prob_pairs = sorted(
            lbl_prob_pairs, key=lambda x: self.class_labels_.index(x[0]))
        return [x[1] for x in sorted_lbl_prob_pairs]

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute classes_.
        """
        return np.array([
            self._format_probas(res)
            for res in self._predict(X, self.num_classes_)
        ], dtype=np.float_)

    def quantize(self, **kwargs):
        """Quantize the model reducing its size and memory footprint.

        Accepts and forwards all keyword arguments defined by Python fasttext's
        ``model.quantize`` method. See Python fasttext docymentation:
        https://github.com/facebookresearch/fastText/tree/master/python#model-object
        """
        self.model.quantize(**kwargs)

    def is_quantized(self):
        """Return true if the inner fasttext model is quantized, else False."""
        return self.model.is_quantized()


class FirstColFtClassifier(FtClassifierABC):
    """An sklearn classifier adapter for fasttext using the first column.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments will be redirected to
        fasttext.train_supervised.
    """

    def _input_col(self, X):
        return np.array(X)[:, 0]


class IdxBasedFtClassifier(FtClassifierABC):
    """An sklearn classifier adapter for fasttext that takes input by index.

    Parameters
    ----------
    input_ix : int
        The index of the text input column for fasttext.
    **kwargs
        Additional keyword arguments will be redirected to
        fasttext.train_supervised.
    """
    def __init__(self, input_ix, **kwargs):
        super().__init__(**kwargs)
        self.input_ix = input_ix

    def _input_col(self, X):
        return np.array(X)[:, self.input_ix]

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        # re-implementation that will preserve ft kwargs
        return {'input_ix': self.input_ix, **self.kwargs}


class FirstObjFtClassifier(FtClassifierABC):
    """An sklearn adapter for fasttext using the first object column as input.

    This classifier assume the X parameter for fit, predict and predict_proba
    is in all cases a pandas.DataFrame object.

    Parameters
    ----------
    **kwargs
        Keyword arguments will be redirected to fasttext.train_supervised.
    """

    def _input_col(self, X):
        input_col_name = None
        for col_name, dtype in X.dtypes.items():
            if dtype == object:
                input_col_name = col_name
                break
        if input_col_name is not None:
            return X[input_col_name]
        raise ValueError("No object dtype column in input param X.")


class ColLblBasedFtClassifier(FtClassifierABC):
    """An sklearn adapter for fasttext taking input by column label.

    This classifier assume the X parameter for fit, predict and predict_proba
    is in all cases a pandas.DataFrame object.

    Parameters
    ----------
    input_col_lbl : str
        The label of the text input column for fasttext.
    **kwargs
        Keyword arguments will be redirected to fasttext.train_supervised.
    """

    def __init__(self, input_col_lbl, **kwargs):
        super().__init__(**kwargs)
        self.input_col_lbl = input_col_lbl

    def _input_col(self, X):
        return X[self.input_col_lbl]

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        # re-implementation that will preserve ft kwargs
        return {'input_col_lbl': self.input_col_lbl, **self.kwargs}
