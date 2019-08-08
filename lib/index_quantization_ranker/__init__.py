from warnings import warn

import numpy as np
import torch
import torch.nn as nn

from lib.quantizer import QuantizationBase, NeuralQuantization
from .wrapper import IndexQuantizationRankerBase
from ..knn import KNNBase, ReorderSearch, FAISSFlatIndex
from ..nn_utils import DISTANCES, CallMethod
from ..utils import process_in_chunks, check_numpy


class NeuralQuantizationRanker(IndexQuantizationRankerBase, KNNBase):
    def __init__(self, base, *, quantizer: QuantizationBase, batch_size=None,
                 device_ids=None, **kwargs):
        """
        A search algorithm that quickly finds nearest neighbors using pre-computed distances
        :param base: dataset to search over, [base_size, vector_dim]
        :param quantizer: quantized search method, typically neural network-based
        :param batch_size: processes base in chunks of this size
        :param quantization_context:
        """
        self.batch_size = batch_size
        self.quantizer = quantizer
        self.device_ids, self.opts = device_ids, kwargs
        if quantizer.training:
            warn("quantizer was asked to process base in training mode (with dropout/bn)")


        if self.device_ids is None:
            base_to_codes = quantizer.get_codes
        else:
            base_to_codes = nn.DataParallel(CallMethod(quantizer, 'get_codes'),
                                            device_ids=device_ids, **kwargs)

        with torch.no_grad():
            base_codes = process_in_chunks(base_to_codes, base,
                                           batch_size=batch_size or len(base))
            # ^-- [base_size, num_codebooks] of uint8
        super().__init__(check_numpy(base_codes).astype(np.uint8))

    def search(self, query, k=1, batch_size=None):
        assert len(query.shape) == 2
        if self.quantizer.training:
            warn("quantizer was asked to search in training mode (with dropout/bn")

        if self.device_ids is None:
            query_to_distances = self.quantizer.get_distances
        else:
            query_to_distances = nn.DataParallel(CallMethod(self.quantizer, 'get_distances'),
                                             device_ids=self.device_ids, **self.opts)

        with torch.no_grad():
            distances_shape = [len(query), self.quantizer.num_codebooks, self.quantizer.codebook_size]
            distances = process_in_chunks(query_to_distances, query,
                                          out=torch.zeros(*distances_shape, dtype=torch.float32, device='cpu'),
                                          batch_size=batch_size or self.batch_size or len(query))
            # ^-- [num_queries, num_codebooks, codebook_size]

        out = np.zeros((distances.shape[0], k), dtype=np.int64)
        super().search(check_numpy(distances), out)
        return out


class UNQSearch(KNNBase):
    def __init__(self, base, *, model:NeuralQuantization, rerank_k=1,
                 reconstructed_distance=DISTANCES['euclidian_squared'],
                 batch_size, reorder_batch_size=None, device_ids=None, **kwargs
                 ):
        apply_model = model if device_ids is None else nn.DataParallel(model, device_ids, **kwargs)
        with torch.no_grad():
            if rerank_k != float('inf'):
                self.knn = NeuralQuantizationRanker(
                    base, quantizer=model, batch_size=batch_size,
                    device_ids=device_ids, **kwargs)
                if rerank_k > 1:
                    # search by quantization, then re-rank the top
                    reconstructed_base = process_in_chunks(apply_model, base, batch_size=batch_size)
                    self.knn = ReorderSearch(
                        reconstructed_base, baseline_search=self.knn, distance=reconstructed_distance,
                        baseline_k=rerank_k, batch_size=reorder_batch_size or batch_size, return_numpy=True)
                    # ^-- note: this code is NOT a time-efficient implementation of reranking search
                    # the efficient implemenation involves searching for rerank_k nearest candidates
                    # with pure NeuralQuantizationRanker and running model.decoder on each candidate

            elif rerank_k == float('inf'):
                # extract data by manually labelling everything
                if reconstructed_distance not in (DISTANCES['euclidian_squared'], DISTANCES['euclidian']):
                    raise NotImplementedError("Fast full reconstruction search is only implemented "
                                              "for euclidian distance")
                reconstructed_base = process_in_chunks(apply_model, base, batch_size=batch_size)
                self.knn = FAISSFlatIndex(reconstructed_base)
            else:
                raise NotImplementedError()

    def search(self, query, k=1, **kwargs):
        return self.knn.search(query, k=k, **kwargs)
