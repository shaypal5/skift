"""fasttext-related utilities."""

import os
from random import randint

from fastText import load_model

# Only create the temp directory if it is needed
TEMP_DIR = None

def get_temp_dir_name():
    # Create the temp dir if it doesn't exist
    if not TEMP_DIR:
        import tempfile
        TEMP_DIR = tempfile.mkdtemp()
    return TEMP_DIR

def temp_dataset_fpath():
    temp_fname = 'temp_ft_trainset_{}.ft'.format(randint(1, 99999))
    return os.path.join(get_temp_dir_name(), temp_fname)


# def dump_df_to_fasttext_format(df, filepath, label_field, text_field):
#     """Dumps the given dataframe to a fasttext-compatible csv file.
#
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The dataframe to dump.
#     filepath : str
#         The fully qualified path to the file to dump.
#     label_field : str
#         The column name of the label field.
#     text_field : str
#         The column name of the label field.
#     """
#     with open(filepath, 'w+') as wfile:
#         for row in df.iterrows():
#             wfile.write('__label__{} {}\n'.format(
#                 row[1][label_field], row[1][text_field]))


def dump_xy_to_fasttext_format(X, y, filepath):
    """Dumps the given X and y matrices  to a fasttext-compatible csv file.

    Parameters
    ----------
    X : array-like, shape = [n_samples]
        The input samples. An array of strings.
    y : array-like, shape = [n_samples]
        The target values. An array of int.
    filepath : str
        The fully qualified path to the file to dump.
    """
    with open(filepath, 'w+') as wfile:
        for text, label in zip(X, y):
            wfile.write('__label__{} {}\n'.format(label, text))


def temp_model_fpath():
    temp_fname = 'temp_ft_model_{}.ft'.format(randint(1, 99999))
    return os.path.join(get_temp_dir_name(), temp_fname)


def python_fasttext_model_to_bytes(model):
    temp_fpath = temp_model_fpath()
    model.save_model(temp_fpath)
    with open(temp_fpath, 'rb') as bfile:
        bytes_obj = bfile.read()
    os.remove(temp_fpath)
    return bytes_obj


def bytes_to_python_fasttext_model(bytes_obj):
    if bytes_obj is None:
        return None
    temp_fpath = temp_model_fpath()
    with open(temp_fpath, 'wb+') as bfile:
        bfile.write(bytes_obj)
    model = load_model(temp_fpath)
    os.remove(temp_fpath)
    return model
