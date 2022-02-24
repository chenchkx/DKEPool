import math
from torch import nn

class MEANNorm(nn.Module):
    def __init__(self, channel, reduction_fixed = False):
        super(MEANNorm, self).__init__()
        if reduction_fixed:
            reduction = 16
        else:
            reduction = 16
            # reduction = int(abs((math.log(channel, 2)+ 1)*2))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.fc(x).view(b, c)
        return y



class ECALayer(nn.Module):

    def __init__(self, channel, auto_k = False):
        super(ECALayer, self).__init__()
        if auto_k:
            t = int(abs((math.log(channel, 2)+ 1)*2))
            k_size = t if t % 2 else t + 1
        else:
            k_size = 15
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x.unsqueeze(1)).squeeze(1)
        y = self.sigmoid(y)

        return y

