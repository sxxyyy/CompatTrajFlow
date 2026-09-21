import gc
import logging
import os
from argparse import Namespace

import torch
from numpy import ndarray
from torch import Generator, Tensor
from torch.amp.grad_scaler import GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.gmvsae.model import GMVSAE
from runner.abstract_runner import AbstractRunner

logger = logging.getLogger(__name__)


class GMVSAERunner(AbstractRunner):
    def __init__(self, args: Namespace, generator: Generator) -> None:
        super().__init__(args, generator)
        self.name = "GMVSAE"
        self.model = GMVSAE(self.num_edges, args, generator)
        self.optimizer = Adam(self.model.parameters(), lr=args.gmvsae_learning_rate)
        self.device = args.device
        self.use_amp = args.use_amp
        self.scaler = GradScaler(args.device.type, enabled=args.use_amp)

    def train(self, dataloader: DataLoader) -> float:
        epoch_loss = 0.0
        for batch_index, mini_batch in enumerate(dataloader):
            padded_paths, mask, lengths, labels = mini_batch
            mini_batch = (
                padded_paths.to(self.device, non_blocking=True),
                mask.to(self.device, non_blocking=True),
                lengths,
                labels.to(self.device, non_blocking=True),
            )
            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                loss: Tensor = self.model(tuple(mini_batch))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
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
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, mini_batch in enumerate(dataloader):
                padded_paths, mask, lengths, labels = mini_batch
                mini_batch = (
                    padded_paths.to(self.device, non_blocking=True),
                    mask.to(self.device, non_blocking=True),
                    lengths,
                    labels.to(self.device, non_blocking=True),
                )
                labels, pred = self.model.test(tuple(mini_batch))
                y_true.append(labels)
                y_pred.append(pred)
                self.log_test_progress(batch_index=i, num_batch=len(dataloader))

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        return y_true, y_pred

    def create_checkpoint_dir(self):
        checkpoint_dir = f"{self.checkpoint_path}/{self.location}/gmvsae"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("Checkpoint directory created at: %s", checkpoint_dir)

    def save_checkpoint(self):
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/gmvsae/model.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info("Model checkpoint saved at: %s", checkpoint_path)

    def load_checkpoint(self):
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/gmvsae/model.pt"
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        logger.info("Model checkpoint loaded from: %s", checkpoint_path)

    def is_checkpoint_exists(self):
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/gmvsae/model.pt"
        exists = os.path.exists(checkpoint_path)
        logger.info("Checkpoint exists: %s", exists)
        return exists

    def free_vram(self):
        del self.model
        del self.optimizer
        if self.use_amp:
            del self.scaler
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("VRAM cleared and garbage collected.")

    def __str__(self) -> str:
        return self.__class__.__name__ + "_" + super().__str__()
