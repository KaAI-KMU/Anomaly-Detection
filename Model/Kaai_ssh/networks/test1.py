#BBox + abstracted Flow -> GRU 
import torch.nn as nn
import torch

class test1_encoder(nn.Module):
    def __init__(self):
        super(test1_encoder, self).__init__()
        Dimension = 8

        
        self.bbox_gru = nn.GRU(input_size = 4, hidden_size = Dimension , num_layers = 1, bias = True, batch_first = True)
        self.flow_gru = nn.GRU(input_size = 2, hidden_size = Dimension , num_layers = 1, bias = True, batch_first = True)

    def forward(self, bbox, flow):
        storage = torch.zeros(flow.shape[0], flow.shape[1], 2).to('cuda')
        for i in range(flow.shape[0]):
            for j in range(flow.shape[1]):
                for k in range(2):
                    storage[i,j,k] = flow[i,j,2,2,k]
        bbox_feature = self.bbox_gru(bbox)
        flow_feature = self.flow_gru(storage)
        feature = (bbox_feature[1].squeeze() + flow_feature[1].squeeze())/2
        return feature
    
class test1_decoder(nn.Module):
    def __init__(self):
        super(test1_decoder, self).__init__()
        Dimension = 8

        self.activation = nn.ReLU()
        
        self.layer = nn.Sequential(
            nn.Linear(Dimension , 4),
            nn.Sigmoid()
        )
    
    def forward(self, feature):
        
        feature = self.activation(feature)
        return self.layer(feature)
    
class test1_autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = test1_encoder()
        self.decoder = test1_decoder()
        
    def forward(self, bbox, flow):
        x = self.encoder(bbox, flow)
        output = self.decoder(x)
        return output