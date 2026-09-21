import gc
import logging
import os
from argparse import Namespace

import numpy as np
import torch
from fastdtw import fastdtw
from numpy import ndarray
from torch import Generator, optim
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.fotraj.model import STAno
from runner.abstract_runner import AbstractRunner

logger = logging.getLogger(__name__)


class FOTrajRunner(AbstractRunner):
    def __init__(self, args: Namespace, generator: Generator) -> None:
        super().__init__(args, generator)
        self.name = "FoTraj"
        self.model = STAno(
            llm_path=args.fotraj_llm_path,
            num_edges=self.num_edges + 1,
            device=args.device,
            drop_out=args.fotraj_dropout,
        )
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(args.device)

        self.optimizer = AdamW(
            self.model.parameters(), lr=args.fotraj_learning_rate, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=2, gamma=0.8
        )
        self.device = args.device
        self.use_amp = True if args.device.type == "cuda" else False
        logger.info("Using mixed precision training: %s", self.use_amp)
        self.scaler = GradScaler(args.device.type, enabled=self.use_amp)
        self.drop_patch_prob = args.fotraj_drop_patch_prob
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.recon_loss = torch.nn.MSELoss(reduction="mean")
        self.anomaly_ratio = args.anomaly_ratio * 100

    def drop_patches(
        self, nodes_tensor, edge_indices_tensor, edge_attrs_tensor, adj_tensor
    ):
        if self.drop_patch_prob > 0:
            nodes_mask = torch.bernoulli(
                (1 - self.drop_patch_prob)
                * torch.ones_like(nodes_tensor, dtype=torch.float)
            )
            nodes_tensor = nodes_tensor * nodes_mask.to(torch.int)

            edge_mask = torch.bernoulli(
                (1 - self.drop_patch_prob)
                * torch.ones_like(edge_attrs_tensor, dtype=torch.float)
            )

            edge_indices_tensor = edge_indices_tensor * edge_mask.unsqueeze(-1).to(
                torch.int
            )
            edge_attrs_tensor = edge_attrs_tensor * edge_mask.to(torch.int)

            adj_tensor = adj_tensor * edge_mask.unsqueeze(1).to(torch.float)

        return nodes_tensor, edge_indices_tensor, edge_attrs_tensor, adj_tensor

    def val(self, dataloader: DataLoader) -> float:
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, mini_batch in enumerate(dataloader):
                (
                    nodes_tensor,
                    edge_indices_tensor,
                    edge_attrs_tensor,
                    nodes_mask_tensor,
                    edge_indices_mask_tensor,
                    edge_attrs_mask_tensor,
                    adj_tensor,
                    adj_mask,
                    _,
                ) = [d.to(self.device) for d in mini_batch]
                last_true_indices = (nodes_mask_tensor.sum(dim=1) - 1).long()

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    mask_nodes = nodes_tensor * nodes_mask_tensor
                    mask_edge_indices = (
                        edge_indices_tensor * edge_indices_mask_tensor.unsqueeze(-1)
                    )
                    mask_edge_attrs = edge_attrs_tensor * edge_attrs_mask_tensor
                    (
                        node_logits,
                        edge_indices_logits,
                        edge_attr_logits,
                        decoded_nodes,
                        decoded_edge_indices,
                        decoded_edge_attrs,
                        embedded_input,
                        embedded_output,
                    ) = self.model(
                        mask_nodes,
                        mask_edge_indices,
                        mask_edge_attrs,
                        adj_tensor,
                        mode="train",
                    )

                    input_nodes_src = mask_nodes[:, 0]
                    output_nodes_src = node_logits[:, 0, :]
                    input_nodes_dst = nodes_tensor[
                        torch.arange(nodes_tensor.shape[0]), last_true_indices
                    ]
                    output_nodes_dst = node_logits[
                        torch.arange(node_logits.shape[0]), last_true_indices
                    ]
                    src_loss = self.ce_loss(output_nodes_src, input_nodes_src.long())
                    dst_loss = self.ce_loss(output_nodes_dst, input_nodes_dst.long())
                    loss1 = 0.1 * src_loss.mean() + 0.9 * dst_loss.mean()

                    node_loss = self.ce_loss(
                        node_logits.permute(0, 2, 1).float(), nodes_tensor.long()
                    )
                    edge_indices_loss = self.ce_loss(
                        edge_indices_logits.permute(0, 1, 3, 2).reshape(
                            -1, edge_indices_logits.shape[2]
                        ),
                        edge_indices_tensor.long().reshape(-1),
                    )
                    edge_attrs_loss = self.ce_loss(
                        edge_attr_logits.permute(0, 2, 1).float(),
                        edge_attrs_tensor.long(),
                    )
                    loss2 = (
                        0.2 * node_loss.mean()
                        + 0.6 * edge_indices_loss.mean()
                        + 0.2 * edge_attrs_loss.mean()
                    )
                    loss3 = self.recon_loss(embedded_output, embedded_input)
                    loss = 0.1 * loss1 + 0.5 * loss2 + 0.4 * loss3

                total_loss.append(loss.item())

        total_loss_avg = float(np.average(total_loss))
        self.model.train()
        logger.info("[Val] Loss=%.4f", total_loss_avg)
        return total_loss_avg

    def early_stopping(self, val_loss: float, *args, **kwargs) -> bool:
        if self.best_loss == float("inf"):
            self.best_loss = val_loss
            self.counter = 0
            return False

        relative_improvement = (
            (self.best_loss - val_loss) / abs(self.best_loss)
            if self.best_loss != 0
            else 0
        )
        if relative_improvement > self.relative_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint()
        else:
            self.counter += 1
            logger.info(
                "No improvement in validation loss for %d times.",
                self.counter,
            )
            if self.counter >= self.patience:
                return True
        return False

    def train(self, dataloader: DataLoader):
        epoch_loss = 0.0
        for batch_index, mini_batch in enumerate(dataloader):
            (
                nodes_tensor,
                edge_indices_tensor,
                edge_attrs_tensor,
                nodes_mask_tensor,
                edge_indices_mask_tensor,
                edge_attrs_mask_tensor,
                adj_tensor,
                adj_mask,
                _,
            ) = [d.to(self.device) for d in mini_batch]
            nodes_tensor, edge_indices_tensor, edge_attrs_tensor, adj_tensor = (
                self.drop_patches(
                    nodes_tensor, edge_indices_tensor, edge_attrs_tensor, adj_tensor
                )
            )
            last_true_indices = (nodes_mask_tensor.sum(dim=1) - 1).long()
            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                mask_nodes = nodes_tensor * nodes_mask_tensor
                mask_edge_indices = (
                    edge_indices_tensor * edge_indices_mask_tensor.unsqueeze(-1)
                )
                mask_edge_attrs = edge_attrs_tensor * edge_attrs_mask_tensor
                (
                    node_logits,
                    edge_indices_logits,
                    edge_attr_logits,
                    decoded_nodes,
                    decoded_edge_indices,
                    decoded_edge_attrs,
                    embedded_input,
                    embedded_output,
                ) = self.model(
                    mask_nodes,
                    mask_edge_indices,
                    mask_edge_attrs,
                    adj_tensor,
                    mode="train",
                )

                input_nodes_src = mask_nodes[:, 0]
                output_nodes_src = node_logits[:, 0, :]
                input_nodes_dst = nodes_tensor[
                    torch.arange(nodes_tensor.shape[0]), last_true_indices
                ]
                output_nodes_dst = node_logits[
                    torch.arange(node_logits.shape[0]), last_true_indices
                ]
                src_loss = self.ce_loss(output_nodes_src, input_nodes_src.long())
                dst_loss = self.ce_loss(output_nodes_dst, input_nodes_dst.long())
                loss1 = 0.1 * src_loss.mean() + 0.9 * dst_loss.mean()
                node_loss = self.ce_loss(
                    node_logits.permute(0, 2, 1).float(), nodes_tensor.long()
                )
                edge_indices_loss = self.ce_loss(
                    edge_indices_logits.permute(0, 1, 3, 2).reshape(
                        -1, edge_indices_logits.shape[2]
                    ),
                    edge_indices_tensor.long().reshape(-1),
                )
                edge_attrs_loss = self.ce_loss(
                    edge_attr_logits.permute(0, 2, 1).float(), edge_attrs_tensor.long()
                )
                loss2 = (
                    0.2 * node_loss.mean()
                    + 0.6 * edge_indices_loss.mean()
                    + 0.2 * edge_attrs_loss.mean()
                )
                loss3 = self.recon_loss(embedded_output, embedded_input)
                loss = 0.1 * loss1 + 0.5 * loss2 + 0.4 * loss3
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
        self.scheduler.step()

    def test(self, dataloader: DataLoader) -> tuple[ndarray, ndarray]:
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, mini_batch in enumerate(dataloader):
                (
                    nodes_tensor,
                    edge_indices_tensor,
                    edge_attrs_tensor,
                    nodes_mask_tensor,
                    edge_indices_mask_tensor,
                    edge_attrs_mask_tensor,
                    adj_tensor,
                    adj_mask,
                    batch_labels,
                ) = [d.to(self.device) for d in mini_batch]
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    mask_nodes = nodes_tensor * nodes_mask_tensor
                    mask_edge_indices = (
                        edge_indices_tensor * edge_indices_mask_tensor.unsqueeze(-1)
                    )
                    mask_edge_attrs = edge_attrs_tensor * edge_attrs_mask_tensor
                    (
                        decoded_nodes,
                        decoded_edge_indices,
                        decoded_edge_attrs,
                        embedded_input,
                    ) = self.model(
                        mask_nodes,
                        mask_edge_indices,
                        mask_edge_attrs,
                        adj_tensor,
                        mode="test",
                    )

                    if (
                        self.anomaly_type == "detour"
                        or self.anomaly_type == "loop"
                        or self.anomaly_type == "switch"
                    ):
                        if isinstance(self.model, torch.nn.DataParallel):
                            embedded_output = self.model.module.embedding_layer(
                                decoded_nodes,
                                decoded_edge_indices,
                                mask_edge_attrs,
                                adj_tensor,
                            )
                        else:
                            embedded_output = self.model.embedding_layer(
                                decoded_nodes,
                                decoded_edge_indices,
                                mask_edge_attrs,
                                adj_tensor,
                            )
                    elif self.anomaly_type in {"time", "time_shift"}:
                        if isinstance(self.model, torch.nn.DataParallel):
                            embedded_output = self.model.module.embedding_layer(
                                mask_nodes,
                                mask_edge_indices,
                                decoded_edge_attrs,
                                adj_tensor,
                            )
                        else:
                            embedded_output = self.model.embedding_layer(
                                mask_nodes,
                                mask_edge_indices,
                                decoded_edge_attrs,
                                adj_tensor,
                            )
                    else:
                        raise ValueError(f"Invalid task={self.anomaly_type}")

                embedded_input_np = (
                    embedded_input.detach().cpu().to(torch.float32).numpy()
                )
                embedded_output_np = (
                    embedded_output.detach().cpu().to(torch.float32).numpy()
                )
                batch_anomaly_scores = []
                for j in range(embedded_input_np.shape[0]):
                    dtw_distance, _ = fastdtw(
                        embedded_input_np[j], embedded_output_np[j]
                    )
                    batch_anomaly_scores.append(dtw_distance)
                y_pred.append(batch_anomaly_scores)

                y_true.append(batch_labels.detach().cpu().numpy())

                self.log_test_progress(batch_index=i, num_batch=len(dataloader))

        y_true = np.concat(y_true)

        y_pred = np.concat(y_pred)
        return y_true, y_pred

    def create_checkpoint_dir(self):
        checkpoint_dir = f"{self.checkpoint_path}/{self.location}/fotraj"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("Checkpoint directory created at: %s", checkpoint_dir)

    def save_checkpoint(self):
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/fotraj/model.pt"
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)
        logger.info("Model checkpoint saved at: %s", checkpoint_path)

    def load_checkpoint(self):
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/fotraj/model.pt"
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        logger.info("Model checkpoint loaded from: %s", checkpoint_path)

    def is_checkpoint_exists(self):
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/fotraj/model.pt"
        exists = os.path.exists(checkpoint_path)
        logger.info("Checkpoint exists: %s", exists)
        return exists

    def free_vram(self):
        del self.model
        del self.optimizer
        del self.scheduler
        if self.use_amp:
            del self.scaler
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("VRAM cleared and garbage collected.")

    def __str__(self) -> str:
        return self.__class__.__name__ + "_" + super().__str__()
