from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import Generator, Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.gmvsae.latent_mixture import LatentGaussianMixture


class GMVSAE(nn.Module):
    def __init__(self, token_size: int, args: Namespace, generator: Generator):
        super().__init__()
        self.padding_idx = 0
        self.bos_idx = 1
        self.num_edges = args.num_edges + 2
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_clusters = args.gmvsae_num_clusters

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

        self.latent_mixture = LatentGaussianMixture(args, generator)

        self.fc_outputs = nn.Linear(args.hidden_size, token_size)

        self.softmax = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, mini_batch: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:

        batch_paths, batch_mask, lengths, _ = mini_batch
        paths_vector = self.embedding(batch_paths)

        packed_paths = nn.utils.rnn.pack_padded_sequence(
            paths_vector, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_z = self.encoder(packed_paths)
        h_z = h_z.squeeze(0)
        z, gaussian_loss, uniform_loss = self.latent_mixture(h_z)

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

        return self.loss(outputs, batch_paths, batch_mask, gaussian_loss, uniform_loss)

    def loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        mask: Tensor,
        gaussian_loss: Tensor,
        uniform_loss: Tensor,
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
        gaussian_loss = torch.mean(gaussian_loss)
        uniform_loss = torch.mean(uniform_loss)

        loss = (
            reconstruction_loss_x
            + gaussian_loss / self.embedding_size
            + uniform_loss / self.num_clusters
        )

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

        batch_bos = torch.full(
            (self.test_batch_size, 1),
            self.bos_idx,
            dtype=torch.long,
        )
        batch_padding = torch.full(
            (self.test_batch_size, 1),
            self.padding_idx,
            dtype=torch.long,
        )
        inputs = torch.cat([batch_bos, batch_paths], dim=1)
        targets = torch.cat([batch_paths, batch_padding], dim=1)
        batch_mask = torch.cat([batch_mask, batch_padding], dim=1)
        mu_c = self.latent_mixture.mu_c
        mu_c_batch = torch.stack([mu_c] * self.test_batch_size, dim=1)

        paths_vector = self.embedding(inputs)

        packed_path = pack_padded_sequence(
            paths_vector, lengths + 1, batch_first=True, enforce_sorted=False
        )

        stack_scores = []
        for mu_c_i in mu_c_batch:
            mu_c_i = mu_c_i.unsqueeze(0)
            outputs, _ = self.decoder(packed_path, mu_c_i)
            outputs, _ = pad_packed_sequence(
                outputs, batch_first=True, padding_value=self.padding_idx
            )
            score = self.get_anomaly_score(outputs, targets, batch_mask)
            stack_scores.append(score)
        max_score = torch.max(torch.vstack(stack_scores), dim=0)[0]

        return label, 1 - max_score

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
