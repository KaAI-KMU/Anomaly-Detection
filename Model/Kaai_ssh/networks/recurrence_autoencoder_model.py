import torch
from torch import nn
from config import model_config as cfg
import numpy as np
from utils.recurrence import rec_plot

class recurrence_flow_autoencoder(nn.Module):
    """Some Information about recurrence_autoencoder"""
    def __init__(self):
        super(recurrence_flow_autoencoder, self).__init__()

        # Flow(2) + BBox(4) + Ego(3) -> 9channel
        # 16 * 16 * 9 Image

        # 16*9 -> 16*16 -> 8*32 -> 4*64 -> 4*128
        self.encoder = nn.Sequential(
            #input = 16 * 16 * 9
            nn.Conv2d(2, 16, 3, padding = 'same'),
            nn.ReLU(),
            # result = 16 * 16 * 16

            #input = 16 * 16 * 16
            nn.Conv2d(16, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 8 * 8 * 32

            #input = 8 * 8 * 32
            nn.Conv2d(32, 64, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 4 * 4 * 64
        
            #input = 4 * 4 * 64
            nn.Conv2d(64, 128, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # result = 2 * 2 * 128
        )

        # 2 * 128 -> 4 * 64 -> 8 * 32 -> 16 * 16 -> 16 * 9
        self.decoder = nn.Sequential(
            # inputs = (Batch, 128, 2, 2)
            nn.Conv2d(128, 64, 3, padding = 'same'),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # result = 4 * 4 * 64

            # input = 4 * 4 * 64
            nn.Conv2d(64, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # result = 8 * 8 * 32

            # input = 8 * 8 * 32
            nn.Conv2d(32, 16, 3, padding = 'same'),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # result = 16 * 16 * 16

            # input = 16 * 16 * 16
            nn.Conv2d(16, 2, 3, padding = 'same', ),
            nn.ReLU()   
        )

    def forward(self, x):
        feature = self.encoder(x)
        result = self.decoder(feature)
        return result
    
class recurrence_bbox_autoencoder(nn.Module):
    """Some Information about recurrence_autoencoder"""
    def __init__(self):
        super(recurrence_bbox_autoencoder, self).__init__()

        # Flow(2) + BBox(4) + Ego(3) -> 9channel
        # 16 * 16 * 9 Image

        # 16*9 -> 16*16 -> 8*32 -> 4*64 -> 4*128
        self.encoder = nn.Sequential(
            #input = 16 * 16 * 9
            nn.Conv2d(4, 16, 3, padding = 'same'),
            nn.ReLU(),
            # result = 16 * 16 * 16

            #input = 16 * 16 * 16
            nn.Conv2d(16, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 8 * 8 * 32

            #input = 8 * 8 * 32
            nn.Conv2d(32, 64, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 4 * 4 * 64
        
            #input = 4 * 4 * 64
            nn.Conv2d(64, 128, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # result = 2 * 2 * 128
        )

        # 2 * 128 -> 4 * 64 -> 8 * 32 -> 16 * 16 -> 16 * 9
        self.decoder = nn.Sequential(
            # inputs = (Batch, 128, 2, 2)
            nn.Conv2d(128, 64, 3, padding = 'same'),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # result = 4 * 4 * 64

            # input = 4 * 4 * 64
            nn.Conv2d(64, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # result = 8 * 8 * 32

            # input = 8 * 8 * 32
            nn.Conv2d(32, 16, 3, padding = 'same'),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # result = 16 * 16 * 16

            # input = 16 * 16 * 16
            nn.Conv2d(16, 4, 3, padding = 'same', ),
            nn.ReLU()   
        )

    def forward(self, x):
        feature = self.encoder(x)
        result = self.decoder(feature)
        return result
    

class recurrence_ego_autoencoder(nn.Module):
    """Some Information about recurrence_autoencoder"""
    def __init__(self):
        super(recurrence_ego_autoencoder, self).__init__()

        # Flow(2) + BBox(4) + Ego(3) -> 9channel
        # 16 * 16 * 9 Image

        # 16*9 -> 16*16 -> 8*32 -> 4*64 -> 4*128
        self.encoder = nn.Sequential(
            #input = 16 * 16 * 9
            nn.Conv2d(3, 16, 3, padding = 'same'),
            nn.ReLU(),
            # result = 16 * 16 * 16

            #input = 16 * 16 * 16
            nn.Conv2d(16, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 8 * 8 * 32

            #input = 8 * 8 * 32
            nn.Conv2d(32, 64, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 4 * 4 * 64
        
            #input = 4 * 4 * 64
            nn.Conv2d(64, 128, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # result = 2 * 2 * 128
        )

        # 2 * 128 -> 4 * 64 -> 8 * 32 -> 16 * 16 -> 16 * 9
        self.decoder = nn.Sequential(
            # inputs = (Batch, 128, 2, 2)
            nn.Conv2d(128, 64, 3, padding = 'same'),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # result = 4 * 4 * 64

            # input = 4 * 4 * 64
            nn.Conv2d(64, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # result = 8 * 8 * 32

            # input = 8 * 8 * 32
            nn.Conv2d(32, 16, 3, padding = 'same'),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # result = 16 * 16 * 16

            # input = 16 * 16 * 16
            nn.Conv2d(16, 3, 3, padding = 'same', ),
            nn.ReLU()   
        )

    def forward(self, x):
        feature = self.encoder(x)
        result = self.decoder(feature)
        return result
    

class recurrence_model(nn.Module):
    """Some Information about recurrence_autoencoder"""
    def __init__(self):
        super(recurrence_model, self).__init__()

        # Flow(2) + BBox(4) + Ego(3) -> 9channel
        # 16 * 16 * 9 Image

        # 16*9 -> 16*16 -> 8*32 -> 4*64 -> 4*128
        self.flow_encoder = nn.Sequential(
            #input = 16 * 16 * 9
            nn.Conv2d(2, 16, 3, padding = 'same'),
            nn.ReLU(),
            # result = 16 * 16 * 16

            #input = 16 * 16 * 16
            nn.Conv2d(16, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 8 * 8 * 32

            #input = 8 * 8 * 32
            nn.Conv2d(32, 64, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 4 * 4 * 64
        
            #input = 4 * 4 * 64
            nn.Conv2d(64, 128, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # result = 2 * 2 * 128
        )

        self.bbox_encoder = nn.Sequential(
            #input = 16 * 16 * 9
            nn.Conv2d(4, 16, 3, padding = 'same'),
            nn.ReLU(),
            # result = 16 * 16 * 16

            #input = 16 * 16 * 16
            nn.Conv2d(16, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 8 * 8 * 32

            #input = 8 * 8 * 32
            nn.Conv2d(32, 64, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 4 * 4 * 64
        
            #input = 4 * 4 * 64
            nn.Conv2d(64, 128, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # result = 2 * 2 * 128
        )

        self.ego_encoder = nn.Sequential(
            #input = 16 * 16 * 9
            nn.Conv2d(3, 16, 3, padding = 'same'),
            nn.ReLU(),
            # result = 16 * 16 * 16

            #input = 16 * 16 * 16
            nn.Conv2d(16, 32, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 8 * 8 * 32

            #input = 8 * 8 * 32
            nn.Conv2d(32, 64, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # result = 4 * 4 * 64
        
            #input = 4 * 4 * 64
            nn.Conv2d(64, 128, 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # result = 2 * 2 * 128
        )

    def forward(self, flow, bbox, ego):

        flow_feature = self.flow_encoder(flow) # 512개
        bbox_feature = self.bbox_encoder(bbox)
        ego_feature = self.ego_encoder(ego)


        return flow_feature + bbox_feature + ego_feature