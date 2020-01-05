
import torch
from torch import nn

import torch.nn.functional as F

from torch.autograd import Variable

class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        batch_size = input.size(1)

        # This bit is very interesting...
        # input = input.t()

        hidden = self._init_hidden(batch_size)

        output, hidden = self.gru(input, hidden)
        # print("GRU hidden output: {hidden.size()}")

        # hidden is equal to output[-1], NO IT IS NOT, using hidden does not work!

        fc_output = self.fc(output[-1])
        # print(f"FC output: {fc_output.size()}")

        fc_output = fc_output.reshape(fc_output.shape[1], -1)
        # fc_output = F.log_softmax(fc_output, dim=1)

        # not 100% sure about the softmax...
        # the results seem to make more sense without it

        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return Variable(hidden)
