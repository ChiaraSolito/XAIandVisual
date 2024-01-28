import torch
import torch.nn.functional as F
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

        # Weights initialization
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.scattering(x)  #  scattering transform
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # classifier
        x = torch.softmax(x, dim=1)  # get probabilities for each class
        return x