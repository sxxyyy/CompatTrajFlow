from argparse import Namespace

import torch
from torch import Generator, Tensor, nn


class LatentGaussianMixture(nn.Module):
    def __init__(self, args: Namespace, generator: Generator) -> None:
        super().__init__()
        self.rnn_dimension = args.hidden_size
        self.batch_size = args.batch_size
        self.num_clusters = args.gmvsae_num_clusters

        self.mu_c = nn.Parameter(
            torch.randn(self.num_clusters, self.rnn_dimension, generator=generator)
        )

        self.log_var_c = torch.zeros([self.num_clusters, self.rnn_dimension])

        # fc that get mu and var from Z and T
        self.fc_mu_z = nn.Linear(self.rnn_dimension, self.rnn_dimension)
        self.fc_log_var_z = nn.Linear(self.rnn_dimension, self.rnn_dimension)

    def forward(self, h_z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the latent Gaussian mixture model.
        Args:
            z (Tensor): Input tensor of shape (batch_size, rnn_dimension).
        Returns:
            Tensor: Output tensor after applying the mixture model.
        """
        mu_z = self.fc_mu_z(h_z)
        log_var_z = self.fc_log_var_z(h_z)
        var = torch.exp(0.5 * log_var_z)

        # Compute the Gaussian mixture output
        z = mu_z + torch.randn_like(mu_z) * torch.sqrt(var)

        stack_z = torch.stack([z] * self.num_clusters, dim=1)
        stack_mu_c = torch.stack([self.mu_c] * self.batch_size, dim=0)
        stack_log_var_c = torch.stack([self.log_var_c] * self.batch_size, dim=0)
        stack_mu_z = torch.stack([mu_z] * self.num_clusters, dim=1)
        stack_log_var_z = torch.stack([log_var_z] * self.num_clusters, dim=1)

        pi_post_logits = -torch.sum(
            (stack_z - stack_mu_c) ** 2 / torch.exp(stack_log_var_c), dim=-1
        )
        pi_post = torch.softmax(pi_post_logits, dim=-1) + 1e-10

        batch_gaussian_loss = 0.5 * torch.sum(
            pi_post
            * torch.mean(
                stack_log_var_c
                + torch.exp(stack_log_var_z) / torch.exp(stack_log_var_c)
                + (stack_mu_z - stack_mu_c) ** 2 / torch.exp(stack_log_var_c),
                dim=-1,
            ),
            dim=-1,
        ) - 0.5 * torch.mean(1 + log_var_z, dim=-1)

        batch_uniform_loss = torch.mean(
            torch.mean(pi_post, 0) * torch.log(torch.mean(pi_post, 0)), dim=-1
        )

        return z, batch_gaussian_loss, batch_uniform_loss
