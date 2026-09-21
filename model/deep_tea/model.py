from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import Tensor, cat, exp, log, mean, nn, square, sum
from torch.nn import Embedding, Module

from .conv_lstm import ConvLSTM


class DeepTea(Module):
    def __init__(self, map_size: tuple[int, int], args: Namespace) -> None:
        super().__init__()
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_edges = args.num_edges
        self.embedding_size = args.embedding_size

        self.map_size = map_size
        self.in_channels = args.in_channels
        self.kernel_size = args.kernel_size
        self.hidden_size = args.hidden_size
        self.num_clusters = args.deeptea_num_clusters

        self.tokens_embedding = Embedding(
            self.num_edges + 1, self.embedding_size, padding_idx=0
        )

        self.map_encoder = ConvLSTM(
            shape=self.map_size,
            in_channels=self.in_channels,
            kernel_size=self.kernel_size,
            num_features=self.hidden_size,
            batch_first=True,
        )
        self.path_encoder = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.path_decoder = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.fc_mu_z = nn.Sequential(
            nn.Linear(in_features=self.map_size[1], out_features=1),
            nn.ReLU(),
            nn.Flatten(2, 3),
            nn.Linear(in_features=self.map_size[0], out_features=1),
            nn.ReLU(),
            nn.Flatten(0, -1),
        )
        self.fc_log_var_z = nn.Sequential(
            nn.Linear(in_features=self.map_size[1], out_features=1),
            nn.ReLU(),
            nn.Flatten(2, 3),
            nn.Linear(in_features=self.map_size[0], out_features=1),
            nn.ReLU(),
            nn.Flatten(0, -1),
        )

        self.w = nn.Parameter(torch.randn(self.embedding_size))
        self.q = nn.Parameter(torch.randn(self.embedding_size))
        self.z_projection = nn.Linear(
            in_features=self.hidden_size, out_features=self.embedding_size
        )

        self.fc_mu_t = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size
        )
        self.fc_log_var_t = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size
        )

        self.mu_c = nn.Parameter(torch.randn(self.num_clusters, self.hidden_size))
        self.log_var_c = torch.zeros(self.num_clusters, self.hidden_size)

        self.fc_output_t = nn.Linear(
            in_features=self.hidden_size, out_features=self.num_edges + 1
        )
        self.softmax_loss = nn.CrossEntropyLoss(reduction="none")

    @staticmethod
    def reparameterization(mu: Tensor, log_var: Tensor):
        """
        Reparameterization trick to sample from a Gaussian distribution
        Args:
            mu (Tensor): Mean of the Gaussian distribution
            log_var (Tensor): Log variance of the Gaussian distribution
        Returns:
            Tensor: Sampled tensor from the Gaussian distribution
        """
        std = exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, map_tokens: Tensor, batch: tuple[Tensor, Tensor, Tensor]
    ) -> Tensor:
        tokens, masks, _ = batch
        embedded_tokens = self.tokens_embedding(tokens)
        _, (z_h, _) = self.map_encoder(map_tokens)
        mu_z = self.fc_mu_z(z_h)
        log_var_z = self.fc_log_var_z(z_h)
        z = self.reparameterization(mu_z, log_var_z)
        z = self.z_projection(z)

        tau = self.w * embedded_tokens + self.q * z
        _, t_h = self.path_encoder(tau)
        t_h = t_h.squeeze(0)  # (batch_size, hidden_size)

        mu_t = self.fc_mu_t(t_h)
        log_var_t = self.fc_log_var_t(t_h)
        t = self.reparameterization(mu_t, log_var_t)  # (batch_size, hidden_size)

        # (batch_size, num_clusters, hidden_size)
        stack_t = torch.stack([t] * self.num_clusters, dim=1)
        stack_mu_c = torch.stack([self.mu_c] * self.batch_size)
        stack_mu_t = torch.stack([mu_t] * self.num_clusters, dim=1)
        stack_log_var_c = torch.stack([self.log_var_c] * self.batch_size)
        stack_log_var_t = torch.stack([log_var_t] * self.num_clusters, dim=1)

        # (batch_size, hidden_size)
        stack_mu_z = torch.stack([mu_z] * self.batch_size)
        stack_log_var_z = torch.stack([log_var_z] * self.batch_size)

        k_given_T_logits = -sum(
            square(stack_t - stack_mu_c) / exp(stack_log_var_c), dim=-1
        )
        k_given_T = F.softmax(k_given_T_logits, dim=-1) + 1e-10

        z_given_T_logits = -sum(square(t - stack_mu_z) / exp(stack_log_var_z), dim=-1)
        z_given_T = F.softmax(z_given_T_logits, dim=-1) + 1e-10

        batch_r_loss = 0.5 * sum(
            k_given_T
            * mean(
                stack_log_var_c
                + exp(stack_log_var_t) / exp(stack_log_var_c)
                + square(stack_mu_t - stack_mu_c) / exp(stack_log_var_c),
                dim=-1,
            ),
            dim=-1,
        ) - 0.5 * mean(1 + log_var_t, dim=-1)

        batch_k_loss = mean(k_given_T, dim=0) * log(mean(k_given_T, dim=0))
        batch_z_loss = mean(z_given_T, dim=0) * log(mean(z_given_T, dim=0))

        batch_zeros = torch.zeros((self.batch_size, 1), dtype=torch.int32)
        targets = torch.cat([tokens, batch_zeros], dim=1)
        tokens = cat([batch_zeros, tokens], dim=1)
        masks = cat([masks, batch_zeros], dim=1)

        embedded_tokens = self.tokens_embedding(tokens)
        tau = self.w * embedded_tokens + self.q * z
        t_h = t_h.unsqueeze(0)
        outputs_t, _ = self.path_decoder(tau, t_h)
        loss = self.loss(
            outputs_t, targets, masks, (batch_r_loss, batch_z_loss, batch_k_loss)
        )
        return loss

    def compute_anomaly_scores(
        self, map_tokens: Tensor, batch: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        tokens, masks, labels = batch

        _, (z_h, _) = self.map_encoder(map_tokens)  # S, num_features, H, Wß
        mu_z = self.fc_mu_z(z_h)
        log_var_z = self.fc_log_var_z(z_h)
        z = self.reparameterization(mu_z, log_var_z)
        z = self.z_projection(z)

        batch_zeros = torch.zeros((self.test_batch_size, 1), dtype=torch.int32)
        targets = torch.cat([tokens, batch_zeros], dim=1)
        tokens = cat([batch_zeros, tokens], dim=1)
        masks = cat([masks, batch_zeros], dim=1)

        embedded_tokens = self.tokens_embedding(tokens)
        tau = self.w * embedded_tokens + self.q * z
        batch_mu_c = torch.stack([self.mu_c] * self.test_batch_size, dim=1)
        result_matrix = []
        for mu_c in batch_mu_c:
            mu_c = mu_c.unsqueeze(0)
            outputs_t, _ = self.path_decoder(tau, mu_c)
            result_matrix.append(self.anomaly_score(outputs_t, targets, masks))
        result_matrix = torch.vstack(result_matrix)
        normal_probability_scores, _ = torch.max(result_matrix, 0)
        anomaly_scores = 1 - normal_probability_scores
        return labels, anomaly_scores

    def anomaly_score(self, outputs: Tensor, targets: Tensor, masks: Tensor):
        masks = masks.to(torch.float32)
        log_sigmoid = nn.LogSigmoid()
        weight = F.embedding(input=targets, weight=self.fc_output_t.weight.data)
        bias = F.embedding(
            input=targets, weight=self.fc_output_t.bias.data.reshape(-1, 1)
        )
        r_given_K = sum(weight * outputs, dim=-1) + bias.squeeze()
        score = sum(
            masks * exp(log_sigmoid(r_given_K)),
            dim=-1,
        ) / sum(masks, dim=-1)
        return score

    def loss(
        self,
        outputs_t: Tensor,
        targets_t: Tensor,
        masks: Tensor,
        latent_loss: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        batch_r_loss, batch_z_loss, batch_k_loss = latent_loss
        masks = masks.to(torch.float32)
        prediction_t = self.fc_output_t(outputs_t)
        prediction_t = prediction_t.transpose(1, 2)
        batch_reconstruction_loss_t = mean(
            masks * self.softmax_loss(prediction_t, targets_t), dim=-1
        )

        reconstruction_loss_t = mean(batch_reconstruction_loss_t)
        r_loss = mean(batch_r_loss)
        z_loss = mean(batch_z_loss)
        k_loss = mean(batch_k_loss)

        return (
            reconstruction_loss_t
            + 1 / self.hidden_size * r_loss
            + z_loss
            + 1 / self.num_clusters * k_loss
        )
