import gc
import logging
import os
from argparse import Namespace
from os.path import exists

import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Generator, GradScaler, Tensor, cat, exp, log, mean, pow, sum
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from model.mst_oatd.model import MSTOATD
from runner.abstract_runner import AbstractRunner

logger = logging.getLogger(__name__)


class MSTOATDRunner(AbstractRunner):
    """
    MSTOATDRunner is a class that implements a runner for the MST-OATD model, which is designed for
    time-causal anomaly detection. It inherits from the AbstractRunner class and provides methods
    for training, testing, and managing the state of the models.

    Attributes:
        interval (int): Time interval used for temporal discretization, determined by the location.
        s_model (MSTOATD): Spatial model instance for anomaly detection.
        t_model (MSTOATD): Temporal model instance for anomaly detection.
        s_optimizer (Optimizer): Optimizer for the spatial model.
        t_optimizer (Optimizer): Optimizer for the temporal model.
        crit (CrossEntropyLoss): Loss function for reconstruction loss.
        detec (CrossEntropyLoss): Loss function for detection, with no reduction.

    Methods:
        __init__(args: Namespace):
            Initializes the MSTOATDRunner instance with the provided arguments, setting up models,
            optimizers, and loss functions.
        train(dataloader: DataLoader):
            Trains the spatial and temporal models using the provided dataloader. Computes the loss,
            performs backpropagation, and updates the model parameters.
        test(test_dataloader: DataLoader):
            Evaluates the models on the test dataset. Computes likelihoods for spatial and temporal
            clusters and returns the anomaly scores.
        gaussian_pdf_log(x, mu, log_var):
            Computes the log of the Gaussian probability density function for a given input.
        gaussian_pdfs_log(x, mus, log_vars) -> Tensor:
            Computes the log of the Gaussian probability density functions for multiple clusters.
        Loss(x_hat, targets, z_mu, z_sigma2_log, z, mode, mask) -> Tensor:
            Computes the total loss, including reconstruction loss, Gaussian loss,
            and category loss, for the given mode (spatial or temporal).
        save_checkpoint():
            Saves the state dictionaries of the spatial and temporal models to checkpoint files.
        load_checkpoint():
            Loads the state dictionaries for the spatial and temporal models from checkpoint files.
    """

    def __init__(self, args: Namespace, generator: Generator) -> None:
        logger.info("Start MST-OSTD Runner")
        super().__init__(args, generator)
        self.name = "MST-OATD"
        self.interval = 10 if args.location == "porto" else 15
        self.mst_num_clusters = args.mst_num_clusters
        self.s_model = MSTOATD(args.num_edges, args.num_edges, args, generator)
        self.t_model = MSTOATD(
            args.num_edges, int(args.num_times / 2 / self.interval), args, generator
        )
        self.s_optimizer = Adam(self.s_model.parameters(), lr=args.s_learning_rate)
        self.t_optimizer = AdamW(self.t_model.parameters(), lr=args.t_learning_rate)
        self.crit = CrossEntropyLoss()
        self.detec = CrossEntropyLoss(reduction="none")
        self.scaler = GradScaler(args.device.type, enabled=args.use_amp)

    def train(self, dataloader: DataLoader) -> float:
        epoch_loss = 0.0
        for batch_index, mini_batch in enumerate(dataloader):
            batch_path, batch_time, batch_tau, batch_mask, lengths, _ = mini_batch
            batch_path, batch_time, batch_tau, batch_mask = map(
                lambda x: x.to(self.device, non_blocking=True),
                [batch_path, batch_time, batch_tau, batch_mask],
            )
            self.s_optimizer.zero_grad()
            self.t_optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                x_hat_s, mu_s, log_var_s, z_s = self.s_model(
                    batch_path,
                    batch_tau,
                    lengths,
                    self.batch_size,
                    "train",
                    -1,
                )
                loss = self.Loss(
                    x_hat_s,
                    batch_path,
                    mu_s.squeeze(0),
                    log_var_s.squeeze(0),
                    z_s.squeeze(0),
                    "s",
                    batch_mask,
                )
                x_hat_t, mu_t, log_var_t, z_t = self.t_model(
                    batch_path,
                    batch_tau,
                    lengths,
                    self.batch_size,
                    "train",
                    -1,
                )
                batch_time = batch_time.to(torch.int64)
                loss += self.Loss(
                    x_hat_t,
                    batch_time,
                    mu_t.squeeze(0),
                    log_var_t.squeeze(0),
                    z_t.squeeze(0),
                    "t",
                    batch_mask,
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.s_optimizer)
            self.scaler.step(self.t_optimizer)
            self.scaler.update()
            epoch_loss += loss.item()
            self.global_step += 1
            self.total_train_loss += loss.item()
            global_avg = self.total_train_loss / self.global_step
            self.log_batch_loss(
                batch_index=batch_index,
                num_batch=len(dataloader),
                loss=global_avg,
            )
        epoch_avg = epoch_loss / len(dataloader)
        return epoch_avg

    def test(self, dataloader: DataLoader) -> tuple[ndarray, ndarray]:
        all_likelihood_s = []
        all_likelihood_t = []
        all_labels = []

        with torch.no_grad():
            for i, mini_batch in enumerate(dataloader):
                batch_path, batch_time, batch_tau, batch_mask, lengths, batch_labels = (
                    mini_batch
                )
                batch_path, batch_time, batch_tau, batch_mask, batch_labels = map(
                    lambda x: x.to(self.device, non_blocking=True),
                    [batch_path, batch_time, batch_tau, batch_mask, batch_labels],
                )
                all_labels.append(batch_labels)

                c_likelihood_s = []
                c_likelihood_t = []

                for c in range(self.mst_num_clusters):
                    output_s, _, _, _ = self.s_model(
                        batch_path,
                        batch_tau,
                        lengths,
                        self.test_batch_size,
                        "test",
                        c,
                    )
                    likelihood_s = -self.detec(
                        output_s.reshape(-1, output_s.shape[-1]),
                        batch_path.reshape(-1),
                    )
                    likelihood_s = exp(
                        sum(
                            batch_mask
                            * (likelihood_s.reshape(self.test_batch_size, -1)),
                            dim=-1,
                        )
                        / sum(batch_mask, 1)
                    )

                    output_t, _, _, _ = self.t_model(
                        batch_path,
                        batch_tau,
                        lengths,
                        self.test_batch_size,
                        "test",
                        c,
                    )
                    batch_time = batch_time.to(torch.int64)
                    likelihood_t = -self.detec(
                        output_t.reshape(-1, output_t.shape[-1]),
                        batch_time.reshape(-1),
                    )
                    likelihood_t = exp(
                        sum(
                            batch_mask
                            * (likelihood_t.reshape(self.test_batch_size, -1)),
                            dim=-1,
                        )
                        / sum(batch_mask, 1)
                    )

                    c_likelihood_s.append(likelihood_s.unsqueeze(0))
                    c_likelihood_t.append(likelihood_t.unsqueeze(0))

                all_likelihood_s.append(cat(c_likelihood_s).max(0)[0])
                all_likelihood_t.append(cat(c_likelihood_t).max(0)[0])
                self.log_test_progress(batch_index=i, num_batch=len(dataloader))

        likelihood_s = cat(all_likelihood_s, dim=0)
        likelihood_t = cat(all_likelihood_t, dim=0)
        result = 1 - likelihood_s * likelihood_t
        all_labels = cat(all_labels, dim=0)

        return all_labels.numpy(force=True), result.numpy(force=True)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_var):
        return -0.5 * (
            sum(
                log(torch.full_like(x, torch.pi) * 2)
                + log_var
                + (x - mu).pow(2) / exp(log_var),
                1,
            )
        )

    def gaussian_pdfs_log(self, x, mus, log_vars) -> Tensor:
        g = []
        for c in range(self.mst_num_clusters):
            g.append(
                self.gaussian_pdf_log(
                    x, mus[c : c + 1, :], log_vars[c : c + 1, :]
                ).view(-1, 1)
            )
        return cat(g, 1)

    def Loss(self, x_hat, targets, z_mu, z_sigma2_log, z, mode, mask) -> Tensor:
        if mode == "s":
            pi = self.s_model.pi_prior
            log_sigma2_c = self.s_model.log_var_prior
            mu_c = self.s_model.mu_prior
        elif mode == "t":
            pi = self.t_model.pi_prior
            log_sigma2_c = self.t_model.log_var_prior
            mu_c = self.t_model.mu_prior
        else:
            raise ValueError

        reconstruction_loss = self.crit(x_hat[mask == 1], targets[mask == 1])

        gaussian_loss = mean(
            mean(
                self.gaussian_pdf_log(z, z_mu, z_sigma2_log).unsqueeze(1)
                - self.gaussian_pdfs_log(z, mu_c, log_sigma2_c),
                dim=1,
            ),
            dim=-1,
        ).mean()

        pi = F.softmax(pi, dim=-1)
        z = z.unsqueeze(1)
        mu_c = mu_c.unsqueeze(0)
        log_sigma2_c = log_sigma2_c.unsqueeze(0)

        logits = -sum(pow(z - mu_c, 2) / exp(log_sigma2_c), dim=-1)
        logits = F.softmax(logits, dim=-1) + 1e-10
        category_loss = mean(sum(logits * (log(logits) - log(pi).unsqueeze(0)), dim=-1))

        loss = (
            reconstruction_loss + gaussian_loss / self.hidden_size + category_loss * 0.1
        )
        return loss

    def create_checkpoint_dir(self):
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/mst_oatd/"
        os.makedirs(checkpoint_path, exist_ok=True)
        logger.info("Creating checkpoint directory for MST-OATD at %s", checkpoint_path)

    def save_checkpoint(self):
        """
        Saves the state dictionaries of the models `s_model` and `t_model` to checkpoint files.
        This method saves the state of two models (`s_model` and `t_model`) to specified file paths
        within the `checkpoint_path` directory. The saved files are named `s_model.pth` and
        `t_model.pth` respectively. logger is used to confirm the save operation and the
        file paths.
        Raises:
            Any exception raised by `torch.save` if the save operation fails.
        """
        s_path = f"{self.checkpoint_path}/{self.location}/mst_oatd/s_model.pt"
        t_path = f"{self.checkpoint_path}/{self.location}/mst_oatd/t_model.pt"
        torch.save(self.s_model.state_dict(), s_path)
        logger.info("Model s_model's state dict saved to %s", s_path)
        torch.save(self.t_model.state_dict(), t_path)
        logger.info("Model t_model's state dict saved to %s", t_path)

    def load_checkpoint(self):
        """
        Loads the state dictionaries for the s_model and t_model from their respective
        checkpoint files and updates the models with the loaded states.
        The checkpoint files are expected to be located in the directory specified by
        `self.checkpoint_path` under the subdirectory `mst_oatd`. The filenames for the
        checkpoints are `s_model.pth` for the s_model and `t_model.pth` for the t_model.
        Logs a message indicating the successful loading of each model's state dictionary.
        Raises:
            FileNotFoundError: If the checkpoint files do not exist at the specified paths.
            RuntimeError: If the state dictionary cannot be loaded into the models.
        """
        s_path = f"{self.checkpoint_path}/{self.location}/mst_oatd/s_model.pt"
        t_path = f"{self.checkpoint_path}/{self.location}/mst_oatd/t_model.pt"
        self.s_model.load_state_dict(torch.load(s_path, self.device))
        logger.info("Model s_model's state dict loaded from %s", s_path)
        self.t_model.load_state_dict(torch.load(t_path, self.device))
        logger.info("Model t_model's state dict loaded from %s", t_path)

    def is_checkpoint_exists(self) -> bool:
        s_path = f"{self.checkpoint_path}/{self.location}/mst_oatd/s_model.pt"
        t_path = f"{self.checkpoint_path}/{self.location}/mst_oatd/t_model.pt"
        return exists(s_path) and exists(t_path)

    def free_vram(self):
        del self.s_model
        del self.t_model
        del self.s_optimizer
        del self.t_optimizer
        if self.use_amp:
            del self.scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("VRAM freed for next training or testing")

    def __str__(self) -> str:
        return self.__class__.__name__ + "_" + super().__str__()
