import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DecoderSD(nn.Module):
    def __init__(self, hidden_size, latent_num) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_linear = nn.Linear(latent_num, hidden_size * 2)

    def forward(self, z):
        """
        Input:
        z (batch_size, latent_size): the latent variable
        ---
        Output:
        hidden (batch, 2, hidden_size)
        """
        hidden = self.hidden_linear(z)
        hidden = hidden.view(hidden.size(0), 2, self.hidden_size).contiguous()

        return hidden


class DecoderRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, layer_num, latent_num, dropout, label_num
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer_num = layer_num
        self.hidden_linear = nn.Linear(latent_num, hidden_size * layer_num)
        self.lstm = nn.GRU(
            input_size, hidden_size, layer_num, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(self.dropout)
        self.label_num = label_num

    def forward(self, z, target, lengths=None, train=True):
        """
        Input:
        z (batch_size, latent_size): the latent variable
        target (batch_size, seq_len, hidden_size): padded sequence tensor
        lengths (batch_size): lengths of the target sequences
        train (bool)
        ---
        Output:
        p_x (batch, seq_len, hidden_size)
        """
        hidden = self.hidden_linear(z)
        hidden = (
            hidden.view(hidden.size(0), self.layer_num, self.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )
        if train:
            if lengths is not None:
                packed_input = pack_padded_sequence(
                    target, lengths, batch_first=True, enforce_sorted=False
                )
            output, hidden = self.lstm(packed_input, hidden)

            if lengths is not None:
                output = pad_packed_sequence(output, batch_first=True)[0]

            p_x = self.dropout(output)

        else:
            outputs = []
            target_len = target.shape[1]
            for i in range(target_len):
                # (batch_size, 1, hidden_size)
                if i == 0:
                    e = target[:, 0, :].unsqueeze(1)
                else:
                    e = outputs[-1].unsqueeze(1)
                # output: (batch_size, 1, hidden_size)
                output, hidden = self.lstm(e, hidden)
                output = output.squeeze(1)
                output = self.dropout(output)
                outputs.append(output)
            outputs = torch.stack(outputs)
            outputs = self.dropout(outputs)
            p_x = outputs.transpose(0, 1).contiguous()

        return p_x


class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        layer_num,
        latent_size,
        dropout,
        bidirectional=True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.lstm = nn.GRU(
            input_size,
            hidden_size // self.num_directions,
            layer_num,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.enc_mu = nn.Linear(layer_num * hidden_size, latent_size)
        self.enc_log_sigma = nn.Linear(layer_num * hidden_size, latent_size)

    def forward(self, input, lengths=None):
        """
        Input:
        input (batch_size, seq_len, hidden_size): padded input sequence tensor
        lengths (batch_size): lengths of input sequences
        ---
        Output:
        q_z: A normal distribution
        mu (batch_size, latent_size): the mean of the normal distribution
        sigma (batch_size, latent_size): the standard deviation of the normal distribution
        """
        if lengths is not None:
            packed_input = pack_padded_sequence(
                input, lengths, batch_first=True, enforce_sorted=False
            )
        output, hidden = self.lstm(packed_input)

        if self.num_directions == 2:
            batch_size, half_hidden = hidden.size(1), hidden.size(2)
            hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)

        mu = self.enc_mu(hidden)
        log_sigma = self.enc_log_sigma(hidden)
        sigma = torch.exp(log_sigma)
        return torch.distributions.Normal(loc=mu, scale=sigma), mu, sigma


class VAE(nn.Module):
    def __init__(
        self, input_size, hidden_size, layer_num, latent_num, dropout, label_num
    ) -> None:
        super().__init__()
        self.enc = EncoderRNN(
            input_size, hidden_size, layer_num, latent_num, dropout, True
        )
        self.dec = DecoderRNN(
            input_size, hidden_size, layer_num, latent_num, dropout, label_num
        )
        self.sd_dec = DecoderSD(hidden_size, latent_num)
        self.label_num = label_num

    def forward(self, src, trg, src_lengths, trg_lengths):
        """
        Input:
        src (batch_size, seq_len, hidden_size): input sequence tensor
        trg (batch_size, seq_len, hidden_size): the target sequence tensor
        src_length (batch_size): lengths of input sequences
        trg_length (batch_size): lengths of target sequences
        ---
        Output:
        kl_loss
        p_x (batch_size, seq_len)
        """
        q_z, mu, sigma = self.enc.forward(src, src_lengths)
        # (batch_size, latent_size)
        z = q_z.rsample()
        # (batch_size, seq_len, hidden_size)
        p_x = self.dec.forward(z, trg[:, :-1], trg_lengths - 1)
        kl_loss = torch.distributions.kl_divergence(
            q_z, torch.distributions.Normal(0, 1.0)
        ).sum(dim=-1)
        sd_p_x = self.sd_dec.forward(z)
        return kl_loss, p_x, sd_p_x
