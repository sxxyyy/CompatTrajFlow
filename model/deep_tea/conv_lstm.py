import torch
from torch import Tensor, nn


class ConvLSTM(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int],
        in_channels: int,
        kernel_size: int,
        num_features: int,
        batch_first: bool = False,
    ) -> None:
        """_summary_

        Args:
            shape (tuple[int, int]): The (height, width) of the hidden state h and cell state c
            in_channels (int): The input channels of convolution 2d
            kernel_size (int): The kernel size of convolution 2d, must be odd
            num_features (int): The number of channels of states, hidden_size in other words
        """
        super().__init__()

        self.shape = shape
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_features = num_features
        self.batch_first = batch_first
        self.conv2d = nn.Conv2d(
            in_channels=in_channels + num_features,
            out_channels=4 * num_features,
            kernel_size=kernel_size,
            stride=1,
            padding=(self.kernel_size - 1) // 2,
        )

    def forward(
        self,
        data: Tensor, # data.shape == (S, B, in_channels, H, W) if batch_first is False
        hidden_state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if self.batch_first:
            data = data.transpose(0, 1)

        # pre_h.shape == pre_c.shape == (B, num_features, H, W)
        if hidden_state is None:
            pre_h = torch.zeros(
                data.size(1),
                self.num_features,
                *self.shape,
                dtype=data.dtype,
                device=data.device
            )
            pre_c = torch.zeros(
                data.size(1),
                self.num_features,
                *self.shape,
                dtype=data.dtype,
                device=data.device
            )
        else:
            pre_h, pre_c = hidden_state

        output_inner = []
        for i in range(data.size(0)):
            x = data[i, ...]
            # input_with_h.shape == (S, input_channel + num_features, H, W)
            input_with_h = torch.cat([x, pre_h], 1)
            # (S, 4 * num_features, H, W)
            output = self.conv2d(input_with_h)
            # i.shape == f.shape == o.shape == g.shape == (S, num_features, H, W)
            output_i, output_f, output_o, output_g = torch.split(
                output, self.num_features, dim=1
            )
            i = torch.sigmoid(output_i)
            f = torch.sigmoid(output_f)
            o = torch.sigmoid(output_o)
            g = torch.tanh(output_g)

            next_c = f * pre_c + i * g
            next_h = o * torch.tanh(next_c)
            output_inner.append(next_h)
            pre_h = next_h
            pre_c = next_c
        output = torch.stack(output_inner)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output[:, -1:], (pre_h, pre_c)
