import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from utils.trajectory import TokenTrajectory


def causal_tad_collate_fn(
    batch: list[tuple[TokenTrajectory, int]],
) -> tuple[list, Tensor, Tensor]:
    paths, labels = zip(*[(t.path, label) for t, label in batch])
    paths = list(paths)
    lengths = torch.tensor([len(p) for p in paths], dtype=torch.long, device="cpu")
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")
    return paths, lengths, labels


def deep_tea_collate_fn(
    batch: list[tuple[TokenTrajectory, int]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Collate function for batching TokenTrajectory objects for DeepTEA. Converts timestamps to tokens.

    Args:
        batch (list[tuple[TokenTrajectory, int]]): Batch of (TokenTrajectory, label) tuples.

    Returns:
        tuple[Tensor, Tensor, Tensor]: Batched path, mask, and labels.
    """
    paths, labels = zip(*[(t.path, label) for t, label in batch])
    paths = [torch.tensor(p, dtype=torch.long, device="cpu") + 1 for p in paths]
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")
    padded_paths = pad_sequence(paths, batch_first=True, padding_value=0)
    mask = (padded_paths != 0).long()
    return padded_paths, mask, labels


def mst_oatd_collate_fn(
    batch: list[tuple[TokenTrajectory, int]],
):
    """Collate function for MST-OATD model. Converts timestamps to tokens and tau features.

    Args:
        batch (list[tuple[TokenTrajectory, int]]): Batch of (TokenTrajectory, label) tuples.

    Returns:
        tuple: Batched path, time tokens, tau features, mask, and labels.
    """

    paths, times, taus, labels = zip(
        *[(t.path, t.time, t.tau, label) for t, label in batch]
    )
    paths = [torch.tensor(p, dtype=torch.long, device="cpu") for p in paths]
    lengths = [len(p) for p in paths]
    times = [torch.tensor(t, dtype=torch.long, device="cpu") for t in times]
    taus = [torch.tensor(tau, dtype=torch.long, device="cpu") for tau in taus]
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")
    padded_paths = pad_sequence(paths, batch_first=True, padding_value=0)
    padded_times = pad_sequence(times, batch_first=True, padding_value=0)
    padded_taus = pad_sequence(taus, batch_first=True, padding_value=0)
    mask = (padded_paths != 0).long()
    return padded_paths, padded_times, padded_taus, mask, lengths, labels


def gmvsae_collate_fn(
    batch: list[tuple[TokenTrajectory, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate function for batching TokenTrajectory objects for GMVSAE. Converts timestamps to tokens.

    Args:
        batch (list[tuple[TokenTrajectory, int]]): Batch of (TokenTrajectory, label) tuples.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Batched path, time tokens, lengths, mask, and labels.
    """
    paths, labels = zip(*[(t.path, label) for t, label in batch])
    paths = [torch.tensor(p, dtype=torch.long, device="cpu") + 2 for p in paths]
    lengths = torch.tensor([len(p) for p in paths], dtype=torch.long, device="cpu")
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")
    padded_paths = pad_sequence(paths, batch_first=True, padding_value=0)
    masks = (padded_paths != 0).long()
    return padded_paths, masks, lengths, labels


def vsae_collate_fn(
    batch: list[tuple[TokenTrajectory, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate function for batching TokenTrajectory objects for VSAE. Converts timestamps to tokens.

    Args:
        batch (list[tuple[TokenTrajectory, int]]): Batch of (TokenTrajectory, label) tuples.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: Batched path, time tokens, lengths, and labels.
    """
    paths, labels = zip(*[(t.path, label) for t, label in batch])
    paths = [torch.tensor(p, dtype=torch.long, device="cpu") + 2 for p in paths]
    lengths = torch.tensor([len(p) for p in paths], dtype=torch.long, device="cpu")
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")
    padded_paths = pad_sequence(paths, batch_first=True, padding_value=0)
    masks = (padded_paths != 0).long()
    return padded_paths, masks, lengths, labels


def fotraj_collate_fn(batch):
    nodes = [item[0] for item in batch]
    edge_indices = [item[1] for item in batch]
    edge_attrs = [item[2] for item in batch]
    adj_matrices = [item[3] for item in batch]
    adj_masks = [item[4] for item in batch]
    labels = [item[5] for item in batch]

    max_nodes = 64
    nodes_mask = []
    for i in range(len(nodes)):
        if nodes[i].size(0) > max_nodes:
            nodes[i] = nodes[i][:max_nodes]
        padding = max_nodes - nodes[i].size(0)
        if padding > 0:
            nodes[i] = torch.cat(
                [
                    nodes[i],
                    torch.zeros(
                        padding,
                        dtype=torch.long,
                        device="cpu",
                    ),
                ]
            )
        mask = torch.cat(
            [
                torch.ones(
                    nodes[i].size(0) - padding,
                    dtype=torch.bool,
                    device="cpu",
                ),
                torch.zeros(
                    padding,
                    dtype=torch.bool,
                    device="cpu",
                ),
            ]
        )
        nodes_mask.append(mask)

    max_edges = 64
    edge_mask = []
    edge_attrs_mask = []
    padded_edge_indices = []
    for i in range(len(edge_attrs)):
        edge_attrs[i] = edge_attrs[i].clamp(max=1599)
        if edge_attrs[i].size(0) > max_edges:
            edge_attrs[i] = edge_attrs[i][:max_edges]
        padding = max_edges - edge_attrs[i].size(0)
        if edge_attrs[i].dim() == 1:
            if padding > 0:
                edge_attrs[i] = torch.cat(
                    [
                        edge_attrs[i],
                        torch.zeros(
                            padding,
                            dtype=torch.long,
                            device="cpu",
                        ),
                    ]
                )
        else:
            if edge_attrs[i].size(1) > 3:
                edge_attrs[i] = edge_attrs[i][:, :3]
            elif edge_attrs[i].size(1) < 3:
                edge_attrs[i] = torch.cat(
                    [
                        edge_attrs[i],
                        torch.zeros(
                            edge_attrs[i].size(0),
                            3 - edge_attrs[i].size(1),
                            dtype=torch.long,
                            device="cpu",
                        ),
                    ],
                    dim=1,
                )
            if padding > 0:
                edge_attrs[i] = torch.cat(
                    [
                        edge_attrs[i],
                        torch.zeros(
                            padding,
                            edge_attrs[i].size(1),
                            dtype=torch.long,
                            device="cpu",
                        ),
                    ],
                    dim=0,
                )

        edge_list = edge_indices[i]
        if len(edge_list) > max_edges:
            edge_list = edge_list[:max_edges]
        padding = max_edges - len(edge_list)
        if padding > 0:
            edge_list = torch.cat(
                [
                    (
                        edge_list.clone().detach()
                        if isinstance(edge_list, torch.Tensor)
                        else torch.tensor(
                            edge_list,
                            dtype=torch.long,
                            device="cpu",
                        )
                    ),
                    torch.full(
                        (padding, 2),
                        0,
                        dtype=torch.long,
                        device="cpu",
                    ),
                ],
                dim=0,
            )

        mask = torch.cat(
            [
                torch.ones(
                    len(edge_list) - padding,
                    dtype=torch.bool,
                    device="cpu",
                ),
                torch.zeros(
                    padding,
                    dtype=torch.bool,
                    device="cpu",
                ),
            ]
        )
        edge_mask.append(mask)

        padded_edge_indices.append(edge_list)
        edge_attrs_mask.append(
            torch.cat(
                [
                    torch.ones(
                        edge_attrs[i].size(0) - padding,
                        dtype=torch.bool,
                        device="cpu",
                    ),
                    torch.zeros(padding, dtype=torch.bool, device="cpu"),
                ]
            )
        )

    nodes_tensor = torch.stack(nodes)
    edge_indices_tensor = torch.stack(padded_edge_indices)
    edge_attrs_tensor = torch.stack(edge_attrs)
    nodes_mask_tensor = torch.stack(nodes_mask)
    edge_mask_tensor = torch.stack(edge_mask)
    edge_attrs_mask_tensor = torch.stack(edge_attrs_mask)
    adj_tensor = torch.stack(adj_matrices)

    adj_masks = torch.stack(adj_masks)
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")

    return (
        nodes_tensor,
        edge_indices_tensor,
        edge_attrs_tensor,
        nodes_mask_tensor,
        edge_mask_tensor,
        edge_attrs_mask_tensor,
        adj_tensor,
        adj_masks,
        labels,
    )
