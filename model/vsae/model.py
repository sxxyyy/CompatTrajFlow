from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VSAE(nn.Module):
    def __init__(self, token_size: int, args: Namespace):
        super().__init__()
        self.padding_idx = 0
        self.bos_idx = 1
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size

        self.embedding = nn.Embedding(
            token_size, args.embedding_size, padding_idx=self.padding_idx
        )
        self.encoder = nn.GRU(
            input_size=args.embedding_size,
            hidden_size=args.hidden_size,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=args.embedding_size,
            hidden_size=args.hidden_size,
            batch_first=True,
        )

        self.fc_log_var_z = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size
        )
        self.fc_mu_z = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size
        )

        self.fc_outputs = nn.Linear(args.hidden_size, token_size)

        self.softmax = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self.log_sigmoid = nn.LogSigmoid()

    def reparameterization(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from a Gaussian distribution.
        Args:
            mu (Tensor): Mean of the Gaussian.
            log_var (Tensor): Log variance of the Gaussian.
        Returns:
            Tensor: Sampled latent variable.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, mini_batch: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:

        batch_paths, batch_mask, lengths, _ = mini_batch
        paths_vector = self.embedding(batch_paths)

        packed_paths = nn.utils.rnn.pack_padded_sequence(
            paths_vector, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_z = self.encoder(packed_paths)
        h_z = h_z.squeeze(0)
        mu_z = self.fc_mu_z(h_z)
        log_var_z = self.fc_log_var_z(h_z)
        z = self.reparameterization(mu_z, log_var_z)

        batch_bos = torch.full(
            (self.batch_size, 1),
            self.bos_idx,
            dtype=torch.long,
            device=batch_paths.device,
        )
        batch_padding = torch.full(
            (self.batch_size, 1),
            self.padding_idx,
            dtype=torch.long,
            device=batch_paths.device,
        )
        batch_paths = torch.cat([batch_bos, batch_paths], dim=1)
        batch_mask = torch.cat([batch_mask, batch_padding], dim=1)

        paths_vector = self.embedding(batch_paths)
        packed_paths = pack_padded_sequence(
            paths_vector, lengths + 1, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.decoder(packed_paths, z.unsqueeze(0))
        outputs, _ = pad_packed_sequence(
            outputs, batch_first=True, padding_value=self.padding_idx
        )

        return self.loss(outputs, batch_paths, mu_z, log_var_z, batch_mask)

    def loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        mu: Tensor,
        log_var: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Compute the loss for the model.

        Args:
            outputs (Tensor): Model outputs.
            targets (Tensor): Target labels.
            mask (Tensor): Mask to ignore padding.

        Returns:
            Tensor: Computed loss.
        """
        reconstruction_loss_x = self.get_reconstruction_loss(outputs, targets, mask)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = reconstruction_loss_x + kld_loss / self.embedding_size

        return loss

    def get_reconstruction_loss(
        self, outputs: Tensor, targets: Tensor, mask: Tensor
    ) -> Tensor:
        """
        Compute the reconstruction loss for the model.

        Args:
            outputs (Tensor): Model outputs.
            targets (Tensor): Target labels.
            mask (Tensor): Mask to ignore padding.

        Returns:
            Tensor: Computed reconstruction loss.
        """
        predict_paths = self.fc_outputs(outputs)
        predict_flat = predict_paths.reshape(-1, predict_paths.size(-1))
        target_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        valid_idx = mask_flat.nonzero(as_tuple=False).squeeze(1)

        return self.softmax(predict_flat[valid_idx], target_flat[valid_idx])

    def test(
        self, mini_batch: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        Test the model on a mini-batch.

        Args:
            mini_batch (tuple[Tensor, Tensor, Tensor, Tensor]): Mini-batch containing paths, times, lengths, and labels.

        Returns:
            Tensor: Anomaly scores for the test batch.
        """
        batch_paths, batch_mask, lengths, label = mini_batch
        paths_vector = self.embedding(batch_paths)

        packed_paths = nn.utils.rnn.pack_padded_sequence(
            paths_vector, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_z = self.encoder(packed_paths)
        h_z = h_z.squeeze(0)
        mu_z = self.fc_mu_z(h_z)
        log_var_z = self.fc_log_var_z(h_z)
        z = self.reparameterization(mu_z, log_var_z)

        batch_bos = torch.full(
            (self.test_batch_size, 1),
            self.bos_idx,
            dtype=torch.long,
            device=batch_paths.device,
        )
        batch_padding = torch.full(
            (self.test_batch_size, 1),
            self.padding_idx,
            dtype=torch.long,
            device=batch_paths.device,
        )
        paths_token = torch.cat([batch_bos, batch_paths], dim=1)
        batch_mask = torch.cat([batch_mask, batch_padding], dim=1)
        targets = torch.cat([batch_paths, batch_padding], dim=1)

        paths_vector = self.embedding(paths_token)
        packed_paths = pack_padded_sequence(
            paths_vector, lengths + 1, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.decoder(packed_paths, z.unsqueeze(0))
        outputs, _ = pad_packed_sequence(
            outputs, batch_first=True, padding_value=self.padding_idx
        )
        scores = self.get_anomaly_score(outputs, targets, batch_mask)

        return label, 1 - scores

    def get_t_given_mu_c(self, targets: Tensor, outputs: Tensor) -> Tensor:
        weight = F.embedding(targets, self.fc_outputs.weight.data)
        bias = F.embedding(targets, torch.reshape(self.fc_outputs.bias.data, (-1, 1)))
        return torch.sum(outputs * weight, dim=-1) + bias.squeeze()

    def get_anomaly_score(
        self, outputs: Tensor, targets: Tensor, masks: Tensor
    ) -> Tensor:
        t_given_mu_c = self.get_t_given_mu_c(targets, outputs)
        score = torch.sum(masks * torch.exp(self.log_sigmoid(t_given_mu_c)), dim=-1)
        return score
