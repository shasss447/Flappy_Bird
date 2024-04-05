import torch
from torch import nn
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, network_type='DQN', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network_type = network_type
        self.layer1 = nn.Linear(input_dim,64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512) 
        self.layer5 = nn.Linear(512, 512)

        if network_type == 'DuelingDQN':
            self.state_values = nn.Linear(512,1)
            self.advantages = nn.Linear(512, output_dim)
        else:
            self.output = nn.Linear(512, output_dim)
        
    def forward(self, x):
          x = F.relu6(self.layer1(x))
          x = F.relu6(self.layer2(x))
          x = F.relu6(self.layer3(x))
          x = F.relu6(self.layer4(x))
          x = F.relu6(self.layer5(x))
          if self.network_type == 'DuelingDQN':
            state_values = self.state_values(x)
            advantages = self.advantages(x)
            output = state_values + (advantages - torch.max((advantages), dim=1, keepdim=True)[0])
            return output
          else:
            return self.output(x)
        