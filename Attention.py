import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_channel, kernel_size):
        super(Attention, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.H = nn.Conv2d(in_channels=hidden_dim,
                           out_channels=attn_channel,
                           kernel_size=kernel_size,
                           padding=self.padding,
                           bias=False).to(device)
        self.W = nn.Conv2d(in_channels=input_dim,
                           out_channels=attn_channel,
                           kernel_size=kernel_size,
                           padding=self.padding,
                           bias=False).to(device)
        self.V = nn.Conv2d(in_channels=attn_channel,
                           out_channels=65,
                           kernel_size=kernel_size,
                           padding=self.padding,
                           bias=False).to(device)

    def forward(self, input_tensor, hidden):
        hid_conv_out = self.H(hidden[0])
        in_conv_out = self.W(input_tensor)
        energy = self.V((hid_conv_out + in_conv_out).tanh())
        return energy
