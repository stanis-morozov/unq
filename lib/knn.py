import faiss
import torch
from functools import partial
from .utils import check_numpy, process_in_chunks, free_memory


class KNNBase:
    def __init__(self, base, **kwargs):
        """ :param base: vectors to search over, [base_size, vector_dim] """
        raise NotImplementedError()

    def search(self, query, k=1):
        """
        :param query: [query_size, vector_dim]
        :param k: nearest neighbors to find
        :return: nearest neighbor indices in base: [query_size, k]
        """
        raise NotImplementedError()


class FAISSFlatIndex(KNNBase):
    def __init__(self, base, use_cuda=torch.cuda.is_available(), device_id=0, temp_memory=True):
        """
        A wrapper for faiss index to quickly find nearest neighbors
        :param base: np array [num_vectors, vector_dim]
        """
        assert len(base.shape) == 2
        dim = base.shape[1]
        self.index_flat = faiss.IndexFlatL2(dim)

        if use_cuda:
            self.res = faiss.StandardGpuResources()
            if not temp_memory:
                self.res.noTempMemory()
            self.index_flat = faiss.index_cpu_to_gpu(self.res, device_id, self.index_flat)

        self.index_flat.add(check_numpy(base))

    def search(self, query, k=1):
        """
        For each query in :query:, finds :k: nearest neighbors in :base: using faiss
        :param query: matrix [num_queries, vector_dim]
        :return: matrix [num_queries, k]
        """
        free_memory()
        assert len(query.shape) == 2
        _, neighbors = self.index_flat.search(check_numpy(query), k)
        free_memory()
        return neighbors


class ReorderSearch:
    def __init__(self, base, *, distance, baseline_search, baseline_k=lambda k: 10 * k,
                 batch_size=None, return_numpy=True, ):
        """
        Finds k nearest neighbors in base using brute force search on top of first_level_search
        :param base: [base_size, vector_dim]
        :param distance: f(x[..., vector_dim], y[..., vector_dim]) -> distance[...], broadcastable
        :param baseline_search: an base search algorithm, typically faster but inexact
        :param baseline_k: a number of neighbors to be queried from baseline_search (before reordering)
            can be an int or a function(output k) -> baseline_k
        :param return_numpy: if True, casts result to a numpy array, if False, keeps torch tensor
        :param batch_size: default batch size; see search docstring for details
        """
        self.base = base
        self.distance = distance
        self.batch_size = batch_size
        self.return_numpy = return_numpy
        self.baseline_search = baseline_search
        self.baseline_k = baseline_k

    def _search(self, query, candidate_ids, k):
        """ private method: full-batch search, use .search() instead """
        candidates = self.base[candidate_ids]
        distances = self.distance(query[:, None, :], candidates)  # [num_queries, base_size]
        chosen_in_candidates = torch.topk(distances, k=k, dim=1, largest=False)[1]  # [num_queries, k]
        batch_range = torch.arange(len(candidates), device=candidates.device)
        return candidate_ids[batch_range[:, None], chosen_in_candidates]

    def search(self, query, k=1, *, baseline_k=None, batch_size=None, **kwargs):
        """
        :param query: [num_queries, vector_dim]
        :param k: find this many nearest neighbors
        :param baseline_k: override for the same parameter in __init__ (defautlts to what was given at __init__)
        :param batch_size: finds nearest neighbors for chunks of this many queries (default = all queries)
            NOTE: this param does not affect baseline_search search
        :param kwargs: keyword args passed to baseline_search search
        :return: matrix [num_queries, k], ids of nearest neighbors in batch
        """
        batch_size = batch_size or self.batch_size or len(query)
        baseline_k = baseline_k or self.baseline_k
        if callable(baseline_k):
            baseline_k = baseline_k(k)

        candidate_ids = self.baseline_search.search(query, k=baseline_k, **kwargs)
        candidate_ids = torch.as_tensor(candidate_ids, device=self.base.device)
        nearest_ids = process_in_chunks(partial(self._search, k=k), query, candidate_ids, batch_size=batch_size)
        if self.return_numpy:
            nearest_ids = check_numpy(nearest_ids)
        return nearest_ids
