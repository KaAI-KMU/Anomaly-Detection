from torch import nn
import numpy as np
class CONV_MODEL(nn.Module):
    """Some Information about CONV_MODEL"""
    def __init__(self, feature):
        super(CONV_MODEL, self).__init__()

        numbers = np.linspace(16, feature, num = 5, dtype = np.int32)

        self.encoder = nn.Sequential(
                        nn.Conv1d(16, numbers[1], 3),
                        nn.ReLU(),
                        nn.Conv1d(numbers[1], numbers[2], 3),
                        nn.ReLU(),
                        nn.Conv1d(numbers[2], numbers[3], 3),
                        nn.ReLU(),
                        nn.Conv1d(numbers[3], numbers[4], 3),
                    )
        self.decoder = nn.Sequential(
                        nn.ConvTranspose1d(numbers[4], numbers[3], 3),
                        nn.ReLU(),
                        nn.ConvTranspose1d(numbers[3], numbers[2], 3),
                        nn.ReLU(),
                        nn.ConvTranspose1d(numbers[2], numbers[1], 3),
                        nn.ReLU(),
                        nn.ConvTranspose1d(numbers[1], 16, 3),
                        nn.Sigmoid()
        )

    def forward(self, data):
        # (Batch, Time, Feature)
        feature = self.encoder(data)
        result = self.decoder(feature)

        return result

class CONV_MODEL_SAD(nn.Module):
    """Some Information about CONV_MODEL_SAD"""
    def __init__(self, feature):

        super(CONV_MODEL_SAD, self).__init__()

        numbers = np.linspace(16, feature, num = 5, dtype = np.int32)
        
        self.encoder = nn.Sequential(
                        nn.Conv1d(16, numbers[1], 3),
                        nn.ReLU(),
                        nn.Conv1d(numbers[1], numbers[2], 3),
                        nn.ReLU(),
                        nn.Conv1d(numbers[2], numbers[3], 3),
                        nn.ReLU(),
                        nn.Conv1d(numbers[3], numbers[4], 3),
                    )
        self.c = None

    def forward(self, data):
        # (Batch, Time, Feature)
        feature = self.encoder(data)

        return feature