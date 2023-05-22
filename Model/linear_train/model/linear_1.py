from torch import nn

class ALL_MODEL(nn.Module):
    """Some Information about ALL_MODEL"""
    def __init__(self, feature):
        super(ALL_MODEL, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(9 * 16, 72),
            nn.ReLU(),
            nn.Linear(72, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, feature)
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 72),
            nn.ReLU(),
            nn.Linear(72, 9 * 16),
            nn.Sigmoid()
        )

    def forward(self, data):
        feature = self.encoder(data)
        result = self.decoder(feature)
        return result
    
class ALL_MODEL_SAD(nn.Module):
    """Some Information about ALL_MODEL_SAD"""
    def __init__(self, feature):
        super(ALL_MODEL_SAD, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(9 * 16, 72),
            nn.ReLU(),
            nn.Linear(72, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, feature)
        )

        self.c = None

    def forward(self, data):
        feature = self.encoder(data)
        return feature
    
