from torch import nn

class CONV_MODEL(nn.Module):
    """Some Information about CONV_MODEL"""
    def __init__(self, feature = 9):
        super(CONV_MODEL, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv1d(16, 25, 3),
                        nn.ReLU(),
                        nn.Conv1d(25, 36, 3),
                        nn.ReLU(),
                        nn.Conv1d(36, 49, 3),
                        nn.ReLU(),
                        nn.Conv1d(49, 64, 3),
                    )
        self.decoder = nn.Sequential(
                        nn.ConvTranspose1d(64, 49, 3),
                        nn.ReLU(),
                        nn.ConvTranspose1d(49, 36, 3),
                        nn.ReLU(),
                        nn.ConvTranspose1d(36, 25, 3),
                        nn.ReLU(),
                        nn.ConvTranspose1d(25, 16, 3),
                        nn.Sigmoid()
        )

    def forward(self, data):
        # (Batch, Time, Feature)
        feature = self.encoder(data)
        result = self.decoder(feature)

        return result

class CONV_MODEL_SAD(nn.Module):
    """Some Information about CONV_MODEL_SAD"""
    def __init__(self, feature = 9):
        super(CONV_MODEL_SAD, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv1d(16, 25, 3),
                        nn.ReLU(),
                        nn.Conv1d(25, 36, 3),
                        nn.ReLU(),
                        nn.Conv1d(36, 49, 3),
                        nn.ReLU(),
                        nn.Conv1d(49, 64, 3),
                    )

        self.c = None

    def forward(self, data):
        # (Batch, Time, Feature)
        feature = self.encoder(data)

        return feature