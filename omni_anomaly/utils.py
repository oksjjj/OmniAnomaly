# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

prefix = "processed"


def save_z(z, filename='z'):
    """Save the sampled z in txt files."""
    for i in range(0, z.shape[1], 20):
        with open(filename + '_' + str(i) + '.txt', 'w') as file:
            for j in range(z.shape[0]):
                for k in range(z.shape[2]):
                    file.write('%f ' % (z[j][i][k]))
                file.write('\n')
    i = z.shape[1] - 1
    with open(filename + '_' + str(i) + '.txt', 'w') as file:
        for j in range(z.shape[0]):
            for k in range(z.shape[2]):
                file.write('%f ' % (z[j][i][k]))
            file.write('\n')


def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    else:
        raise ValueError('unknown dataset ' + str(dataset))


def default_pot_level(dataset):
    """
    Paper Appendix B POT low quantile by dataset.

    SMAP 0.07, MSL 0.01,
    SMD machine-1-* 0.005, machine-2-* 0.0025, machine-3-* 0.0001.
    """
    name = str(dataset)
    if name == 'SMAP':
        return 0.07
    if name == 'MSL':
        return 0.01
    if name.startswith('machine-1-'):
        return 0.005
    if name.startswith('machine-2-'):
        return 0.0025
    if name.startswith('machine-3-'):
        return 0.0001
    if name.startswith('machine'):
        return 0.005  # fallback for unexpected SMD names
    raise ValueError('unknown dataset ' + name)


def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True,
             do_preprocess=True, train_start=0, test_start=0):
    """
    Load data from pkl files.

    Returns:
        ((train_data, None), (test_data, test_label))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size

    if print_log:
        print('load data of:', dataset)
        print("train: ", train_start, train_end)
        print("test: ", test_start, test_end)

    x_dim = get_data_dim(dataset)
    with open(os.path.join(prefix, dataset + '_train.pkl'), "rb") as f:
        train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]

    try:
        with open(os.path.join(prefix, dataset + '_test.pkl'), "rb") as f:
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
    except (KeyError, FileNotFoundError):
        test_data = None

    try:
        with open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb") as f:
            test_label = pickle.load(f).reshape((-1,))[test_start:test_end]
    except (KeyError, FileNotFoundError):
        test_label = None

    if do_preprocess:
        train_data = preprocess(train_data)
        if test_data is not None:
            test_data = preprocess(test_data)

    if print_log:
        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
        if test_label is not None:
            print("test set label shape: ", test_label.shape)

    return (train_data, None), (test_data, test_label)


def preprocess(df):
    """Return MinMax-normalized data."""
    df = np.asarray(df, dtype=np.float32)

    if df.ndim == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(np.isnan(df)):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num(df)

    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')
    return df


def minibatch_slices_iterator(length, batch_size, ignore_incomplete_batch=False):
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """Mini-batch iterator for sliding windows."""

    def __init__(self, array_size, window_size, batch_size, excludes=None,
                 shuffle=False, ignore_incomplete_batch=False):
        if window_size < 1:
            raise ValueError('`window_size` must be at least 1')
        if array_size < window_size:
            raise ValueError('`array_size` must be at least as large as `window_size`')

        if excludes is not None:
            excludes = np.asarray(excludes, dtype=bool)
            if excludes.shape != (array_size,):
                raise ValueError(
                    f'The shape of `excludes` is expected to be {(array_size,)}, '
                    f'but got {excludes.shape}'
                )

        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=bool)
        mask[: window_size - 1] = False

        if excludes is not None:
            where_excludes = np.where(excludes)[0]
            for k in range(1, window_size):
                also_excludes = where_excludes + k
                also_excludes = also_excludes[also_excludes < array_size]
                mask[also_excludes] = False

        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])
        self._offsets = np.arange(-window_size + 1, 1)
        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty')

        if self._shuffle:
            np.random.shuffle(self._indices)

        for s in minibatch_slices_iterator(
                length=len(self._indices),
                batch_size=self._batch_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] if a.ndim == 1 else a[idx, :] for a in arrays)
