from torch import nn


class EntropyApprox(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(EntropyApprox, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size // 4),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size // 4, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, 1))

    def forward(self, input):
        return self.net(input)
