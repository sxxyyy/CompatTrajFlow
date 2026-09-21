import gc
import logging
import os
import pickle
from argparse import Namespace
from copy import deepcopy
from os.path import exists

import numpy as np
import torch
from numpy import ndarray
from torch import Generator, GradScaler, LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.causal_tad.model import CausalTADModel
from runner.abstract_runner import AbstractRunner
from utils.edge_remapper import get_or_build_edge_mapping

logger = logging.getLogger(__name__)


class CausalTADRunner(AbstractRunner):
    def __init__(self, args: Namespace, generator: Generator) -> None:
        super().__init__(args, generator)
        self.name = "CausalTAD"
        self.model = CausalTADModel(
            self.hidden_size, self.hidden_size, self.device, 1, args.num_edges + 3
        )
        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.causal_tad_learning_rate,
            weight_decay=args.causal_tad_weight_decay,
        )
        self.scaler = GradScaler(self.device.type, enabled=args.use_amp)
        self.adjacency_dict = self.load_adjacency_dict()
        self.num_edges = args.num_edges + 3  # Adjust for pad, end and start tokens

    def sample_subgraph(self, edge_list: Tensor) -> LongTensor:
        points = list(set(edge_list.view(-1).tolist()))
        sample_neighs = []
        for point in points:
            neighs = self.adjacency_dict.get(point, [])
            sample_neighs.append(set(neighs))

        column_indices = [n for sample_neigh in sample_neighs for n in sample_neigh]
        row_indices = [
            points[i] for i in range(len(points)) for _ in range(len(sample_neighs[i]))
        ]
        sub_graph_edges = torch.LongTensor([row_indices, column_indices], device="cpu")
        return sub_graph_edges

    def load_adjacency_dict(self) -> dict:
        """Load the adjacency dictionary required for CausalTAD from a file."""
        logger.info("Loading CausalTAD required adjacency dictionary")
        with open(f"data/{self.location}/raw/adj_dict.pkl", "rb") as f:
            raw_adj_dict = pickle.load(f)

        edge_mapping = get_or_build_edge_mapping(self.location)

        adj_dict = {}
        for raw_u, mapped_u in edge_mapping.items():
            if raw_u in raw_adj_dict:
                mapped_neighs = []
                for raw_v in raw_adj_dict[raw_u]:
                    if raw_v in edge_mapping:
                        mapped_neighs.append(edge_mapping[raw_v])
                if mapped_neighs:
                    adj_dict[mapped_u] = list(set(mapped_neighs))

        return adj_dict

    def train(self, dataloader: DataLoader):
        epoch_loss = 0.0
        for batch_index, mini_batch in enumerate(dataloader):
            paths, src_lengths, _ = mini_batch
            trg = deepcopy(paths)
            # Add start and end tokens
            for target in trg:
                target.insert(0, self.num_edges - 3)
                target.append(self.num_edges - 2)

            trg_lengths = LongTensor(
                [length + 2 for length in src_lengths], device="cpu"
            )

            src = [torch.tensor(path, device="cpu") for path in paths]
            trg = [torch.tensor(target, device="cpu") for target in trg]

            src = pad_sequence(src, batch_first=True, padding_value=self.num_edges - 1)
            trg = pad_sequence(trg, batch_first=True, padding_value=self.num_edges - 1)

            sub_graph_edges = self.sample_subgraph(src)

            src = src.to(self.device)
            trg = trg.to(self.device)
            sub_graph_edges = sub_graph_edges.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                nll_loss, kl_loss, confidence, sd_loss = self.model.forward(
                    src, trg, sub_graph_edges, src_lengths, trg_lengths
                )

                nll_loss = nll_loss.sum(dim=-1).mean()
                confidence = confidence.sum(dim=-1).mean()
                loss = nll_loss + kl_loss.mean() + confidence + sd_loss
                loss = loss.mean()
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
        all_labels = []
        order_prob = []
        with torch.no_grad():
            for batch_index, mini_batch in enumerate(dataloader):
                paths, src_lengths, labels = mini_batch
                trg = deepcopy(paths)
                # Add start and end tokens
                for target in trg:
                    target.insert(0, self.num_edges - 3)
                    target.append(self.num_edges - 2)
                trg_lengths = LongTensor(
                    [length + 2 for length in src_lengths], device="cpu"
                )

                src = [torch.tensor(path, device="cpu") for path in paths]
                trg = [torch.tensor(target, device="cpu") for target in trg]

                src = pad_sequence(
                    src, batch_first=True, padding_value=self.num_edges - 1
                )
                trg = pad_sequence(
                    trg, batch_first=True, padding_value=self.num_edges - 1
                )

                sub_graph_edges = self.sample_subgraph(src)

                src = src.to(self.device)
                trg = trg.to(self.device)
                sub_graph_edges = sub_graph_edges.to(self.device)

                nll_loss, _, confidence, _ = self.model.forward(
                    src, trg, sub_graph_edges, src_lengths, trg_lengths
                )
                prob = nll_loss.cpu().detach().tolist()
                confidence_list = confidence.cpu().detach().tolist()
                src_lengths = src_lengths.cpu().detach().tolist()
                for j, item in enumerate(prob):
                    order_prob.append([src_lengths[j], item, confidence_list[j]])
                all_labels.append(labels)
                self.log_test_progress(batch_index, len(dataloader))

            new_output = []
            for length, nll, mask in order_prob:
                nll = np.array(nll[: int(length) + 1])
                mask = np.array(mask[: int(length)] + [0])
                nll = nll - mask * 0.1
                new_output.append(sum(nll))

            y_true = torch.cat(all_labels).cpu().numpy()
            y_pred = -np.array(new_output)
        return y_true, y_pred

    def is_checkpoint_exists(self):
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/causal_tad/model.pt"
        return exists(checkpoint_path)

    def create_checkpoint_dir(self):
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/causal_tad/"
        os.makedirs(checkpoint_path, exist_ok=True)
        logger.info(
            "Creating checkpoint directory for CausalTAD at %s", checkpoint_path
        )

    def save_checkpoint(self):
        """
        Save the model state dict to the checkpoint path.
        """
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/causal_tad/model.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info("CausalTAD model checkpoint saved at %s", checkpoint_path)

    def load_checkpoint(self):
        """
        Load the model state dict from the checkpoint path.
        """
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/causal_tad/model.pt"
        self.model.load_state_dict(torch.load(checkpoint_path, self.device))
        logger.info("CausalTAD model checkpoint loaded from %s", checkpoint_path)

    def free_vram(self):
        del self.model
        del self.optimizer
        if self.use_amp:
            del self.scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("VRAM freed for next training or testing")

    def __str__(self) -> str:
        return self.__class__.__name__ + "_" + super().__str__()
