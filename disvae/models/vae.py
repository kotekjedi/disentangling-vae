"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

from disvae.utils.initialization import weights_init
from .encoders import get_encoder
from .decoders import get_decoder

DECODERS = ["Burgess", "Objectives"]
ENCODERS = ["Burgess", "Objectives"]


def init_specific_model(encoder_type, decoder_type, img_size, latent_dim, objectives):

    encoder_type = encoder_type.lower().capitalize()
    decoder_type = decoder_type.lower().capitalize()

    if encoder_type not in ENCODERS:
        err = "Unkown encoder_type={}. Possible values: {}"
        raise ValueError(err.format(encoder_type, ENCODERS))

    if decoder_type not in DECODERS:
        err = "Unkown decoder_type={}. Possible values: {}"
        raise ValueError(err.format(encoder_type, DECODERS))

    encoder = get_encoder(encoder_type)
    decoder = get_decoder(decoder_type)
    if objectives > 0:
        model = VAE(img_size, encoder, decoder, latent_dim, objectives)
    else:
        model = VAE(img_size, encoder, decoder, latent_dim)
    model.encoder_type = encoder_type  # store to help reloading
    model.decoder_type = decoder_type
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim, objectives=None):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64], [128, 128], [128, 107]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.objectives = objectives
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(self.img_size, self.latent_dim, self.objectives)
        self.decoder = decoder(self.img_size, self.latent_dim, self.objectives)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct, objectives = self.decoder(latent_sample)

        return reconstruct, latent_dist, latent_sample, objectives

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample
