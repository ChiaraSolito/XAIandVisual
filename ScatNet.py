import torch
import torch.nn.functional as F
import torch.optim
from kymatio.torch import Scattering2D

'''
ScatNet 2D
'''


class ScatNet2D(torch.nn.Module):
    def __init__(self, input_channels: int, scattering: Scattering2D, num_classes: int):
        super(ScatNet2D, self).__init__()

        self.input_channels = input_channels
        self.scattering = scattering
        self.num_classes = num_classes

        # linear with size [K*25*25, 1024],
        # linear with size [1024, 1024],
        # linear with size [1024,3]
        # All the linear layer are followed by ReLU activation
        # 32: grandezza immagine dopo scattering
        self.lin = torch.nn.Linear(self.input_channels * 32 * 32 * 3, 576)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(576, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, self.num_classes)
        )

        # Weights initialization
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.scattering(x)  # scattering transform
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        x = self.classifier(x)  

        return x