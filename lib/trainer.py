import os
import inspect
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import UNQModel
from .utils import check_numpy, get_latest_file
from .knn import FAISSFlatIndex
from .nn_utils import OneCycleSchedule, DISTANCES, training_mode, gumbel_softmax
from .quantizer import compute_penalties
from tensorboardX import SummaryWriter


class Trainer(nn.Module):
    def __init__(self, *, model, experiment_name=None,
                 Loss=None, loss_opts=None, optimizer=None,
                 LearnedSimilaritySearch, SimilaritySearch=FAISSFlatIndex,
                 NegativeSimilaritySearch=None, warm_start=False, verbose=False, max_norm=None, drop_large_grads=False,
                 device_ids=None, output_device=None, batch_dim=0, **kwargs):
        """
        A class that handles model training, checkpointing and evaluation
        :type model: NeuralQuantizationModel
        :param experiment_name: a path where all logs and checkpoints are saved
        :type Loss: module that computes loss function, see LossBase
        :param loss_opts: a dictionary of parameters for self.compute_loss
        :param Optimizer: function(parameters) -> optimizer
        :param SimilaritySearch: knn engine to be used for recall evaluation
        :param device_ids: if given, performs data-parallel training on these device ids
        :param output_device: gathers loss on this device id
        :param batch_dim: if device_ids is specified, batch tensors will be split between devices along this axis
        :param max_norm: if gradient global norm exceeds this value, clips gradient by it's norm
        """
        super().__init__()
        self.model = model
        self.loss = (Loss or AutoencoderLoss)(model, **loss_opts)
        if device_ids is not None:
            self.loss = nn.DataParallel(self.loss, device_ids, output_device=output_device, dim=batch_dim)
        self.opt = optimizer or OneCycleSchedule(torch.optim.Adam(model.parameters(), amsgrad=True), **kwargs)
        self.NegativeSimilaritySearch = NegativeSimilaritySearch or LearnedSimilaritySearch
        self.LearnedSimilaritySearch = LearnedSimilaritySearch
        self.SimilaritySearch = SimilaritySearch
        self.verbose = verbose
        self.max_norm = max_norm
        self.drop_large_grads = drop_large_grads
        self.drops = 0
        self.step = 0

        if experiment_name is None:
            experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
            if self.verbose:
                print('using automatic experiment name: ' + experiment_name)
        self.experiment_path = os.path.join('logs/', experiment_name)
        if not warm_start and experiment_name != 'debug':
            assert not os.path.exists(self.experiment_path), 'experiment {} already exists'.format(experiment_name)
        self.writer = SummaryWriter(self.experiment_path, comment=experiment_name)
        if warm_start:
            self.load_checkpoint()

        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for arg in args:
            self.writer.add_text(str(arg), str(values[arg]), global_step=self.step)

    def train_on_batch(self, *batch, prefix='train/', **kwargs):
        self.opt.zero_grad()
        with training_mode(self.model, self.loss, is_train=True):
            metrics = self.loss(*batch, **kwargs)
        metrics['loss'].mean().backward()
        if self.max_norm is not None:
            metrics['grad_norm'] = torch.as_tensor(torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm))
            
            if self.drop_large_grads and metrics['grad_norm'] > self.max_norm:
                print("Grad dropped!")
                self.drops += 1
                for metric in metrics:
                    self.writer.add_scalar(prefix + metric, metrics[metric].mean().item(), self.step)
                return metrics
                
        self.opt.step()
        self.step += 1
        for metric in metrics:
            self.writer.add_scalar(prefix + metric, metrics[metric].mean().item(), self.step)
        return metrics

    def evaluate_recall(self, base, query, k=1, prefix='dev/', **kwargs):
        """ Computes average recall @ k """
        with torch.no_grad(), training_mode(self.model, is_train=False):
            reference_indices = self.SimilaritySearch(base, **kwargs).search(query, k=1)
            predicted_indices = self.LearnedSimilaritySearch(base, **kwargs).search(query, k=k)
            predicted_indices, reference_indices = map(check_numpy, (predicted_indices, reference_indices))
            recall = np.equal(predicted_indices, reference_indices).any(-1).mean()
        self.writer.add_scalar('{}recall@{}'.format(prefix, k), recall, self.step)
        return recall

    def save_checkpoint(self, tag=None, path=None, mkdir=True, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            tag = self.step
        if path is None:
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(OrderedDict([
            ('model', self.state_dict(**kwargs)),
            ('opt', self.opt.state_dict()),
            ('step', self.step)
        ]), path)
        if self.verbose:
            print("Saved " + path)
        return path

    def load_checkpoint(self, tag=None, path=None, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            path = get_latest_file(os.path.join(self.experiment_path, 'checkpoint_*.pth'))
        elif tag is not None and path is None:
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint['model'], **kwargs)
        self.opt.load_state_dict(checkpoint['opt'])
        self.step = int(checkpoint['step'])
        if self.verbose:
            print('Loaded ' + path)
        return self

    def get_true_nearest_ids(self, base, *, k, exclude_self=True):
        """ returns indices of k nearest neighbors for each vector in original space """
        if self.verbose:
            print(end="Computing ground truth neighbors... ", flush=True)
        k = k or self.positive_neighbors
        with torch.no_grad():
            train_neighbors_index = self.SimilaritySearch(base).search(
                base, k=k + int(exclude_self))[:, int(exclude_self):]
            original_neighbors_index = torch.as_tensor(train_neighbors_index, device=base.device)
        if self.verbose:
            print(end="Done\n", flush=True)
        return original_neighbors_index

    def get_negative_ids(self, base, positive_ids, *, k, skip_k=0):
        """
        returns indices of top-k nearest neighbors in learned space excluding positive_ids
        :param base: float matrix [num_vectors, vector_dim]
        :param positive_ids: int matrix [num_vectors, num_positive_neighbors]
        :param k: number of negative samples for each vector
        :param skip_k: excludes this many nearest indices from nearest ids (used for recall@10/100)
        """
        if self.verbose:
            print(end="Computing negative candidates... ", flush=True)
        assert base.shape[0] == positive_ids.shape[0]
        num_vectors, k_positives = positive_ids.shape
        k_total = k + skip_k + k_positives + 1

        with torch.no_grad():
            with training_mode(self.model, is_train=False):
                learned_nearest_ids = self.NegativeSimilaritySearch(base).search(base, k=k_total)
                learned_nearest_ids = torch.as_tensor(learned_nearest_ids, device=base.device)
            # ^-- [base_size, k_total]

            idendity_ids = torch.arange(len(positive_ids), device=positive_ids.device)[:, None]  # [batch_size, 1]
            forbidden_ids = torch.cat([idendity_ids, positive_ids], dim=1)
            # ^-- [base_size, 1 + k_positives]

            negative_mask = (learned_nearest_ids[..., None] != forbidden_ids[..., None, :]).all(-1)
            # ^-- [base_size, k_total]
            negative_ii, negative_jj = negative_mask.nonzero().t()
            negative_values = learned_nearest_ids[negative_ii, negative_jj]
            # shape(negative_ii, negative_jj, negative_values) = [sum(negative_mask)] (1D)

            # beginning of each row in negative_ii
            slices = torch.cat([torch.zeros_like(negative_ii[:1]),
                                1 + (negative_ii[1:] != negative_ii[:-1]).nonzero()[:, 0]])
            # ^--[base_size]

            # column indices of negative samples
            squashed_negative_jj = torch.arange(len(negative_jj), device=negative_jj.device) - slices[negative_ii]

            # a matrix with nearest elements in learned_nearest_index
            # that are NOT PRESENT in positive_ids for that element
            new_negative_ix = torch.stack((negative_ii, squashed_negative_jj), dim=0)
            negative_ids = torch.sparse_coo_tensor(new_negative_ix, negative_values,
                                                   size=learned_nearest_ids.shape).to_dense()[:, skip_k: k + skip_k]
            # ^--[base_size, k + skip_k]

        if self.verbose:
            print(end="Done\n", flush=True)
        return negative_ids


class LossBase(nn.Module):
    """ A module that implements loss function. compatible with nn.DataParallel """
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.opts = kwargs

    def forward(self, *batch, **kwargs):
        metrics = self.compute_loss(self.model, *batch, **dict(self.opts, **kwargs))
        for key in metrics:
            if not torch.is_tensor(metrics[key]):
                continue
            if len(metrics[key].shape) == 0:
                metrics[key] = torch.unsqueeze(metrics[key], 0)
        return metrics

    @staticmethod
    def compute_loss(model, *batch, **kwargs):
        """
        Something that takes data batch and outputs a dictionary of tensors
        All tensors should be of fixed shape except for (optionally) batch dim
        """
        raise NotImplementedError()


class AutoencoderLoss(LossBase):
    @staticmethod
    def compute_loss(model: UNQModel, x_batch, *, distance=DISTANCES['euclidian_squared'], **kwargs):
        """ A simple loss that minimizes the :distance: between input vector and model output """
        x_reconstructed, activations = model.forward(x_batch, return_intermediate_values=True)
        reconstruction_loss = distance(x_batch, x_reconstructed).mean()
        penalties = compute_penalties(activations['logits'], **kwargs)
        metrics = dict(reconstruction_loss=reconstruction_loss, **penalties)
        metrics['loss'] = reconstruction_loss + penalties['reg']
        return metrics


class TripletLoss(LossBase):
    @staticmethod
    def compute_loss(model: UNQModel,
                     x_batch, x_positives=None, x_negatives=None, *, hard_codes=True,
                     reconstruction_coeff=0.0, reconstruction_distance=DISTANCES['euclidian_squared'],
                     triplet_coeff=0.0, triplet_delta=0.0, eps=1e-6, **kwargs):
        assert (x_positives is not None and x_negatives is not None) or triplet_coeff == 0

        # compute logits with manually applied temperatures for performance reasons
        x_reconstructed, activations = model.forward(x_batch, return_intermediate_values=True)
        # ^-- all: [batch_size, num_codebooks, codebook_size]

        metrics = dict(loss=torch.zeros([], device=x_batch.device))
        if reconstruction_coeff != 0:
            reconstruction_distances = reconstruction_distance(x_batch, x_reconstructed)
            reconstruction_loss = reconstruction_distances.mean()
            metrics['reconstruction_loss'] = reconstruction_loss
            metrics['loss'] += reconstruction_loss * reconstruction_coeff

        if triplet_coeff != 0:
            distances_to_codes = activations['distances_to_codes']

            pos_codes = gumbel_softmax(
                model.compute_logits(x_positives),
                noise=0.0, hard=hard_codes, dim=-1
            )
            pos_distances = (pos_codes * distances_to_codes).sum(dim=[-1, -2])

            neg_codes = gumbel_softmax(
                model.compute_logits(x_negatives),
                noise=0.0, hard=hard_codes, dim=-1
            )
            neg_distances = (neg_codes * distances_to_codes).sum(dim=[-1, -2])

            triplet_loss = F.relu(triplet_delta + pos_distances - neg_distances).mean()
            metrics['triplet_loss'] = triplet_loss
            metrics['loss'] += triplet_coeff * triplet_loss

        # regularizers
        for key, value in compute_penalties(activations['logits'], **kwargs).items():
            assert key not in metrics
            metrics[key] = value.to(device=metrics['loss'].device)
        metrics['loss'] += metrics['reg']
        return metrics
