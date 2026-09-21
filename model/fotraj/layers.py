import math

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dense_to_sparse

def sample_from_logits(logits, top_k=50, temperature=1.0):
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(dim=-1, index=indices, src=values)
    probs = torch.softmax(mask, dim=-1)
    sampled = torch.multinomial(probs, 1).squeeze(-1)
    return sampled


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=8059):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class GraphEncoder(nn.Module):
    def __init__(self, node_vocab_size, edge_indices_size, edge_attr_size, hidden_dim):
        super(GraphEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Embedding(node_vocab_size, hidden_dim)
        self.edge_indices_embedding = nn.Embedding(edge_indices_size, hidden_dim)
        self.edge_attr_embedding = nn.Embedding(edge_attr_size, hidden_dim)

        self.positional_embedding = PositionalEmbedding(hidden_dim, max_len=node_vocab_size)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.gnn_layer = SAGEConv(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes, edge_indices, edge_attr, adj_matrices):
        batch_size, seq_len = nodes.size()
        node_embeddings = self.node_embedding(nodes)
        position_embeddings = self.positional_embedding(node_embeddings)
        node_embeddings += position_embeddings

        edge_index, _ = dense_to_sparse(adj_matrices)
        data = Data(x=node_embeddings.view(-1, self.hidden_dim), edge_index=edge_index)

        node_embeddings = self.gnn_layer(data.x, data.edge_index)
        node_embeddings = self.proj(node_embeddings)
        node_embeddings = node_embeddings.view(batch_size, seq_len, self.hidden_dim)
        node_embeddings = self.layer_norm(node_embeddings)

        gating_input = node_embeddings.view(-1, self.hidden_dim)
        gating_scores = self.gate_mlp(gating_input)
        gating_scores = gating_scores.view(batch_size, seq_len, 1)
        node_embeddings = node_embeddings * gating_scores

        edge_indices_embeddings_1 = self.edge_indices_embedding(edge_indices[..., 0])
        edge_indices_embeddings_2 = self.edge_indices_embedding(edge_indices[..., 1])
        edge_indices_embeddings = edge_indices_embeddings_1 + edge_indices_embeddings_2

        edge_attr_embeddings = self.edge_attr_embedding(edge_attr)
        x = node_embeddings + edge_indices_embeddings + edge_attr_embeddings

        x = self.layer_norm(x)
        return x


class GraphDecoder(nn.Module):
    def __init__(self, node_vocab_size, edge_indices_size, edge_attr_size, hidden_dim):
        super(GraphDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_decoder = nn.Linear(hidden_dim, node_vocab_size, bias=False)
        self.node_embedding = nn.Embedding(node_vocab_size, hidden_dim)
        self.edge_index_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_indices_size * 2)
        )
        self.edge_attr_decoder = nn.Linear(hidden_dim, edge_attr_size)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout_layer = nn.Dropout(p=0.1)

    def forward(self, llama_output, mode='train'):
        x = llama_output
        batch_size, seq_len, hidden_dim = x.size()
        x = self.layer_norm(x)

        node_logits = self.node_decoder(x)
        decoded_nodes = torch.argmax(node_logits, dim=-1)
        decoded_node_embeds = self.node_embedding(decoded_nodes)
        combined_embeds = torch.cat([x, decoded_node_embeds], dim=-1)
        edge_indices_logits_raw = self.edge_index_mlp(combined_embeds)
        edge_indices_logits = edge_indices_logits_raw.view(batch_size, seq_len, -1, 2)

        edge_attr_logits = self.edge_attr_decoder(x)

        if mode == 'train':
            decoded_edge_indices = torch.argmax(edge_indices_logits, dim=-2)
            decoded_edge_attrs = torch.argmax(edge_attr_logits, dim=-1)
            return node_logits, edge_indices_logits, edge_attr_logits, decoded_nodes, decoded_edge_indices, decoded_edge_attrs
        elif mode in ['eval', 'test']:
            node_logits_2d = node_logits.view(-1, node_logits.size(-1))
            sampled_nodes_2d = sample_from_logits(node_logits_2d)
            decoded_nodes = sampled_nodes_2d.view(batch_size, seq_len)

            edge_indices_logits_src = edge_indices_logits[..., 0]
            edge_indices_logits_dst = edge_indices_logits[..., 1]
            edge_indices_logits_src_2d = edge_indices_logits_src.view(-1, edge_indices_logits_src.size(-1))
            edge_indices_logits_dst_2d = edge_indices_logits_dst.view(-1, edge_indices_logits_dst.size(-1))
            sampled_edge_src_2d = sample_from_logits(edge_indices_logits_src_2d)
            sampled_edge_dst_2d = sample_from_logits(edge_indices_logits_dst_2d)

            sampled_edge_src = sampled_edge_src_2d.view(batch_size, seq_len)
            sampled_edge_dst = sampled_edge_dst_2d.view(batch_size, seq_len)
            decoded_edge_indices = torch.stack([sampled_edge_src, sampled_edge_dst], dim=-1)

            edge_attr_logits_2d = edge_attr_logits.view(-1, edge_attr_logits.size(-1))
            sampled_edge_attr_2d = sample_from_logits(edge_attr_logits_2d)
            decoded_edge_attrs = sampled_edge_attr_2d.view(batch_size, seq_len)
            decoded_edge_attrs = torch.minimum(decoded_edge_attrs, torch.tensor(600).to(decoded_edge_attrs.device))

            return decoded_nodes, decoded_edge_indices, decoded_edge_attrs
        else:
            raise ValueError(f"Unknown mode: {mode}")

