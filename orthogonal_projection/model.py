import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, dim=512, mul=4, p : torch.float32 = 0.3 ):
        super().__init__
        self.net = nn.Sequential(
            nn.Linear(dim, dim//mul),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(dim//mul, dim)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight) # Bias의 초기화는 언제쯤 ?

    def forward(self, x):
        x = self.net(x)
        return x
    

class MobileNet(nn.Module):
    pass