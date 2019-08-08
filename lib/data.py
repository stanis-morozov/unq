import os
import warnings

import numpy as np
import torch
import random
from .utils import fvecs_read, download
import os.path as osp


class Dataset:

    def __init__(self, dataset, data_path='./data', normalize=False, random_state=50, **kwargs):
        """
        Dataset is a bunch of tensors with all the learning and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATSETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param random_state: global random seed for an experiment
        :param normalize: if True, divides all data points by an average l2 norm of train_vectors
        :param kwargs: depending on the dataset, you may select train size, test size or other params
            If dataset is not in DATASETS, provide three keys: train_vectors, test_vectors and query_vectors

        """
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)

        if dataset in DATASETS:
            data_dict = DATASETS[dataset](osp.join(data_path, dataset), **kwargs)
        else:
            assert all(key in kwargs for key in ('train_vectors', 'test_vectors', 'query_vectors'))
            data_dict = kwargs

        self.train_vectors = torch.as_tensor(data_dict['train_vectors'])
        self.test_vectors = torch.as_tensor(data_dict['test_vectors'])
        self.query_vectors = torch.as_tensor(data_dict['query_vectors'])
        assert self.train_vectors.shape[1] == self.test_vectors.shape[1] == self.query_vectors.shape[1]
        self.vector_dim = self.train_vectors.shape[1]

        mean_norm = self.train_vectors.norm(p=2, dim=-1).mean().item()
        if normalize:
            self.train_vectors /= mean_norm
            self.test_vectors /= mean_norm
            self.query_vectors /= mean_norm
        else:
            if mean_norm < 0.1 or mean_norm > 10.0:
                warnings.warn("Mean train_vectors norm is {}, consider normalizing")


def fetch_DEEP1M(path, train_size=5 * 10 ** 5, test_size=10 ** 6, ):
    base_path = osp.join(path, 'deep_base1M.fvecs')
    learn_path = osp.join(path, 'deep_learn500k.fvecs')
    query_path = osp.join(path, 'deep_query10k.fvecs')
    if not all(os.path.exists(fname) for fname in (base_path, learn_path, query_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/e23sdc3twwn9syk/deep_base1M.fvecs?dl=1", base_path,
                 chunk_size=4 * 1024 ** 2)
        download("https://www.dropbox.com/s/4i0c5o8jzvuloxy/deep_learn500k.fvecs?dl=1", learn_path,
                 chunk_size=4 * 1024 ** 2)
        download("https://www.dropbox.com/s/5z087cxqh61n144/deep_query10k.fvecs?dl=1", query_path)
    return dict(
        train_vectors=fvecs_read(learn_path)[:train_size],
        test_vectors=fvecs_read(base_path)[:test_size],
        query_vectors=fvecs_read(query_path)
    )


def fetch_BIGANN1M(path, train_size=None, test_size=None):
    base_path = osp.join(path, 'bigann_base1M.fvecs')
    learn_path = osp.join(path, 'bigann_learn500k.fvecs')
    query_path = osp.join(path, 'bigann_query10k.fvecs')
    if not all(os.path.exists(fname) for fname in (base_path, learn_path, query_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/zcnvsy7mlogj4g0/bigann_base1M.fvecs?dl=1", base_path,
                 chunk_size=4 * 1024 ** 2)
        download("https://www.dropbox.com/s/dviygi2zhk57p9m/bigann_learn500k.fvecs?dl=1", learn_path,
                 chunk_size=4 * 1024 ** 2)
        download("https://www.dropbox.com/s/is6anxwon6g5bpe/bigann_query10k.fvecs?dl=1", query_path)
    return dict(
        train_vectors=fvecs_read(learn_path)[:train_size],
        test_vectors=fvecs_read(base_path)[:test_size],
        query_vectors=fvecs_read(query_path)
    )


DATASETS = {
    'DEEP1M': fetch_DEEP1M,
    'BIGANN1M': fetch_BIGANN1M,
}
