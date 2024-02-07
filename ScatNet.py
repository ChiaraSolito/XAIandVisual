import torch
import torch.nn
import torch.optim
from kymatio.torch import Scattering2D

'''
ScatNet 2D
'''


class ScatNet2D(torch.nn.Module):
    def __init__(self, input_channels: int, scattering: Scattering2D):
        super(ScatNet2D, self).__init__()

        self.input_channels = input_channels
        self.scattering = scattering

        # linear with size [K*25*25, 1024],
        # linear with size [1024, 1024],
        # linear with size [1024,3]
        # All the linear layer are followed by ReLU activation

        self.classifier = torch.nn.Sequential(
            # torch.nn.Linear((self.input_channels * 25 * 25, 1024),
            torch.nn.Linear(248832, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2),
            torch.nn.ReLU()
        )

        # self.fc1 = nn.Linear(576,256)
        # self.drop = nn.Dropout(p=0.1)
        # self.fc2 = nn.Linear(256,32)
        # self.fc3 = nn.Linear(32,self.num_classes)

        # Weights initialization
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.scattering(x)  # scattering transform
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  
        
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.drop(x)
        # x = self.fc3(x)

        return x