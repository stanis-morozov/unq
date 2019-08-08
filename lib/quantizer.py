from contextlib import contextmanager

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_utils import Lambda, GumbelSoftmax
from .utils import check_numpy


class QuantizationBase(nn.Module):
    def get_codes(self, x):
        """
        :param x: vectors to be quantized [..., vector_dim]
        :return: integer codes of shape [..., num_codebooks]
        """
        raise NotImplementedError()

    def get_distances(self, x):
        """
        :param x: query vectors to estimate distances
        :return: distances to each code of shape [..., num_codebooks, codebook_size]
        """
        raise NotImplementedError()


class NeuralQuantization(QuantizationBase):
    def __init__(self, vector_dim, num_codebooks=8, codebook_size=256, initial_entropy=1.0, key_dim=None,
                 decouple_temperatures=True, encoder=None, decoder=None, init_codes_with_data=True, **kwargs):
        """
        Multi-codebook quantization that supports nonlinear encoder/decoder transformations
        :param vector_dim: size of vectors to be quantized
        :param num_codebooks: the number of discrete spaces to quantize data vectors into
        :param codebook_size: the number of vectors in each codebook
        :param initial_entropy: starting entropy value for data-aware initialization
        :param init_codes_with_data: if specified, codes will be initialized with random data vectors
        :param encoder: if specified, a torch.nn.Module that maps data vectors
            from [..., *anything] to [..., num_codebooks, key_dim]
        :param decoder: if specified, a torch.nn.Module that maps reconstructed vectrors
            from [..., num_codebooks, codebook_size] to [..., *anything]
        :param decouple_temperatures: if True, does not apply temperatures to logits when computing quantized distances
        """
        super().__init__()
        key_dim = key_dim or vector_dim
        self.num_codebooks, self.codebook_size = num_codebooks, codebook_size
        self.decouple_temperatures = decouple_temperatures
        self.encoder = encoder or nn.Sequential(
            nn.Linear(vector_dim, num_codebooks * key_dim),
            Lambda(lambda x: x.view(*x.shape[:-1], num_codebooks, key_dim)),
        )
        self.codebook = nn.Parameter(torch.randn(num_codebooks, codebook_size, key_dim))
        self.decoder = decoder or nn.Sequential(
            Lambda(lambda x: x.view(*x.shape[:-2], -1)),
            nn.Linear(num_codebooks * codebook_size, vector_dim),
        )
        self.log_temperatures = nn.Parameter(data=torch.zeros(num_codebooks) * float('nan'), requires_grad=True)
        self.initial_entropy, self.init_codes_with_data = initial_entropy, init_codes_with_data
        self.gumbel_softmax = GumbelSoftmax(**kwargs)

    def compute_logits(self, x, add_temperatures=True):
        """ Computes logits for code probabilities [batch_size, num_codebooks, codebook_size] """
        assert len(x.shape) >= 2, "x should be of shape [..., vector_dim]"
        if len(x.shape) > 2:
            flat_logits = self.compute_logits(x.view(-1, x.shape[-1]), add_temperatures=add_temperatures)
            return flat_logits.view(*x.shape[:-1], self.num_codebooks, self.codebook_size)

        # einsum: [b]atch_size, [n]um_codebooks, [c]odebook_size, [v]ector_dim
        logits = torch.einsum('bnd,ncd->bnc', self.encoder(x), self.codebook)

        if add_temperatures:
            if not self.is_initialized(): self.initialize(x)
            logits *= torch.exp(-self.log_temperatures[:, None])
        return logits

    def forward(self, x, return_intermediate_values=False):
        """ quantizes x into one-hot codes and restores original data """
        if not self.is_initialized(): self.initialize(x)
        logits_raw = self.compute_logits(x, add_temperatures=False)
        logits = logits_raw * torch.exp(-self.log_temperatures[:, None])
        codes = self.gumbel_softmax(logits, dim=-1)  # [..., num_codebooks, codebook_size]
        x_reco = self.decoder(codes)

        if return_intermediate_values:
            distances_to_codes = - (logits_raw if self.decouple_temperatures else logits)
            return x_reco, dict(x=x, logits=logits, codes=codes, x_reco=x_reco,
                                distances_to_codes=distances_to_codes)
        else:
            return x_reco

    def get_codes(self, x):
        """ encodes x into uint8 codes[..., num_codebooks] """
        return self.compute_logits(x, add_temperatures=False).argmax(dim=-1)

    def get_distances(self, x):
        # Note: this quantizer uses the fact that logits = - distances
        return - self.compute_logits(x, add_temperatures=not self.decouple_temperatures)

    def is_initialized(self):
        # note: can't set this as property because https://github.com/pytorch/pytorch/issues/13981
        return check_numpy(torch.isfinite(self.log_temperatures.data)).all()

    def initialize(self, x):
        """ Initialize codes and log_temperatures given data """
        with torch.no_grad():
            if self.init_codes_with_data:
                chosen_ix = torch.randint(0, x.shape[0], size=[self.codebook_size * self.num_codebooks], device=x.device)
                chunk_ix = torch.arange(self.codebook_size * self.num_codebooks, device=x.device) // self.codebook_size
                initial_keys = self.encoder(x)[chosen_ix, chunk_ix].view(*self.codebook.shape).contiguous()
                self.codebook.data[:] = initial_keys

            base_logits = self.compute_logits(
                x, add_temperatures=False).view(-1, self.num_codebooks, self.codebook_size)
            # ^-- [batch_size, num_codebooks, codebook_size]

            log_temperatures = torch.tensor([
                fit_log_temperature(codebook_logits, target_entropy=self.initial_entropy, tolerance=1e-2)
                for codebook_logits in check_numpy(base_logits).transpose(1, 0, 2)
            ], device=x.device, dtype=x.dtype)
            self.log_temperatures.data[:] = log_temperatures


def fit_log_temperature(logits, target_entropy=1.0, tolerance=1e-6, max_steps=100,
                        lower_bound=math.log(1e-9), upper_bound=math.log(1e9)):
    """
    Returns a temperature s.t. the average entropy equals mean_entropy (uses bin-search)
    :param logits: unnormalized log-probabilities, [batch_size, num_outcomes]
    :param target_entropy: target entropy to fit
    :returns: temperature (scalar) such that
        probs = exp(logits / temperature) / sum(exp(logits / temperature), axis=-1)
        - mean(sum(probs * log(probs), axis=-1)) \approx mean_entropy
    """
    assert isinstance(logits, np.ndarray)
    assert logits.ndim == 2
    assert 0 < target_entropy < np.log(logits.shape[-1])
    assert lower_bound < upper_bound
    assert np.isfinite(lower_bound) and np.isfinite(upper_bound)

    log_tau = (lower_bound + upper_bound) / 2.0

    for i in range(max_steps):
        # check temperature at the geometric mean between min and max values
        log_tau = (lower_bound + upper_bound) / 2.0
        tau_entropy = _entropy_with_logits(logits, log_tau)

        if abs(tau_entropy - target_entropy) < tolerance:
            break
        elif tau_entropy > target_entropy:
            upper_bound = log_tau
        else:
            lower_bound = log_tau
    return log_tau


def _entropy_with_logits(logits, log_tau=0.0, axis=-1):
    logits = np.copy(logits)
    logits -= np.max(logits, axis, keepdims=True)
    logits *= np.exp(-log_tau)
    exps = np.exp(logits)
    sum_exp = exps.sum(axis)
    entropy_values = np.log(sum_exp) - (logits * exps).sum(axis) / sum_exp
    return np.mean(entropy_values)


def compute_penalties(logits, individual_entropy_coeff=0.0, allowed_entropy=0.0, global_entropy_coeff=0.0,
                      cv_coeff=0.0, square_cv=True, eps=1e-9):
    """
    Computes typical regularizers for gumbel-softmax quantization
    Regularization is of slight help when performing hard quantization, but it isn't critical
    :param logits: tensor [batch_size, ..., codebook_size]
    :param individual_entropy_coeff: penalizes mean individual entropy
    :param allowed_entropy: does not penalize individual_entropy if it is below this value
    :param cv_coeff: penalizes squared coefficient of variation
    :param global_entropy_coeff: coefficient for entropy of mean probabilities over batch
        this value should typically be negative (e.g. -1), works similar to cv_coeff
    """
    counters = dict(reg=torch.tensor(0.0, dtype=torch.float32, device=logits.device))
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    # [batch_size, ..., codebook_size]

    if individual_entropy_coeff != 0:
        individual_entropy_values = - torch.sum(p * logp, dim=-1)
        clipped_entropy = F.relu(allowed_entropy - individual_entropy_values + eps).mean()
        individual_entropy = (individual_entropy_values.mean() - clipped_entropy).detach() + clipped_entropy

        counters['reg'] += individual_entropy_coeff * individual_entropy
        counters['individual_entropy'] = individual_entropy

    if global_entropy_coeff != 0:
        global_p = torch.mean(p, dim=0)  # [..., codebook_size]
        global_logp = torch.logsumexp(logp, dim=0) - np.log(float(logp.shape[0]))  # [..., codebook_size]
        global_entropy = - torch.sum(global_p * global_logp, dim=-1).mean()
        counters['reg'] += global_entropy_coeff * global_entropy
        counters['global_entropy'] = global_entropy

    if cv_coeff != 0:
        load = torch.mean(p, dim=0)  # [..., codebook_size]
        mean = load.mean()
        variance = torch.mean((load - mean) ** 2)
        if square_cv:
            counters['cv_squared'] = variance / (mean ** 2 + eps)
            counters['reg'] += cv_coeff * counters['cv_squared']
        else:
            counters['cv'] = torch.sqrt(variance + eps) / (mean + eps)
            counters['reg'] += cv_coeff * counters['cv']

    return counters
