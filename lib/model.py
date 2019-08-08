import torch.nn as nn
from .nn_utils import Feedforward, Lambda, BatchNorm
from .quantizer import NeuralQuantization


class UNQModel(NeuralQuantization):
    def __init__(self, input_dim, *, hidden_dim=1024, encoder_layers=2, decoder_layers=2,
                 bottleneck_dim=256, num_codebooks=8, codebook_size=256, share_codewords=True,
                 **kwargs):
        """ NeuralQuantization with extra layers on both encoder and decoder sides """
        encoder = nn.Sequential(
            Feedforward(input_dim, hidden_dim, num_layers=encoder_layers, **kwargs),
            nn.Linear(hidden_dim, num_codebooks * bottleneck_dim),
            Lambda(lambda x: x.view(*x.shape[:-1], num_codebooks, bottleneck_dim))
        )
        
        decoder = nn.Sequential()
        decoder.add_module('reshape', Lambda(lambda x: x.view(*x.shape[:-2], -1)))
        if not share_codewords:
            decoder.add_module('embed', nn.Linear(num_codebooks * codebook_size, bottleneck_dim))
        else:
            decoder.add_module('embed', Lambda(lambda x: x @ self.codebook.to(device=x.device).view(num_codebooks * codebook_size, bottleneck_dim)))
        decoder.add_module('batchnorm', BatchNorm(bottleneck_dim))
        decoder.add_module('ffn', Feedforward(bottleneck_dim, hidden_dim, num_layers=decoder_layers, **kwargs))
        decoder.add_module('final', nn.Linear(hidden_dim, input_dim))
        
        super().__init__(
            input_dim, num_codebooks, codebook_size, key_dim=bottleneck_dim,
            encoder=encoder, decoder=decoder, **kwargs
        )
