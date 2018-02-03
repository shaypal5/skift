"""fasttext-related utilities."""


def dump_df_to_fasttext_format(df, filepath, label_field, text_field):
    """Dumps the given dataframe to a fasttext-compatible csv file.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to dump.
    filepath : str
        The fully qualified path to the file to dump.
    label_field : str
        The column name of the label field.
    text_field : str
        The column name of the label field.
    """
    with open(filepath, 'w+') as wfile:
        for row in df.iterrows():
            wfile.write('__label__{} {}\n'.format(
                row[1][label_field], row[1][text_field]))


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
