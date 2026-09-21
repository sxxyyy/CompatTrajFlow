import logging

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .layers import GraphDecoder, GraphEncoder

logger = logging.getLogger(__name__)


class STAno(nn.Module):
    def __init__(self, llm_path, num_edges, device, drop_out):
        super(STAno, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.model = AutoModelForCausalLM.from_pretrained(llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        self.embedding_layer = GraphEncoder(
            node_vocab_size=num_edges,
            edge_indices_size=num_edges,
            edge_attr_size=1600,
            hidden_dim=self.model.config.hidden_size,
        )
        self.decoder_layer = GraphDecoder(
            node_vocab_size=num_edges,
            edge_indices_size=num_edges,
            edge_attr_size=1600,
            hidden_dim=self.model.config.hidden_size,
        )

        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, nodes, edge_indices, edge_attrs, adj_metrics, mode="train"):
        embedded_x = self.embedding_layer(nodes, edge_indices, edge_attrs, adj_metrics)
        outputs = self.model(inputs_embeds=embedded_x, output_hidden_states=True)
        logits = outputs.hidden_states[-1]
        if mode == "train":
            logits = self.dropout(logits)
            (
                node_logits,
                edge_indices_logits,
                edge_attr_logits,
                decoded_nodes,
                decoded_edge_indices,
                decoded_edge_attrs,
            ) = self.decoder_layer(logits, mode=mode)
            embedded_outputs = self.embedding_layer(
                decoded_nodes, decoded_edge_indices, decoded_edge_attrs, adj_metrics
            )
            return (
                node_logits,
                edge_indices_logits,
                edge_attr_logits,
                decoded_nodes,
                decoded_edge_indices,
                decoded_edge_attrs,
                embedded_x,
                embedded_outputs,
            )
        else:
            decoded_nodes, decoded_edge_indices, decoded_edge_attrs = (
                self.decoder_layer(logits, mode=mode)
            )
            return decoded_nodes, decoded_edge_indices, decoded_edge_attrs, embedded_x
