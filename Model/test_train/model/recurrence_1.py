import torch
from torch import nn
from config.model_config import *

class flow_ae(nn.Module):
    """Some Information about flow_ae"""
    def __init__(self, feature):
        super(flow_ae, self).__init__()

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(flow_input_channel, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, feature, 2),
            
        )
        self.flow_decoder = nn.Sequential(
            nn.ConvTranspose2d(feature, 256, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(4, flow_input_channel, 3),
            nn.Sigmoid() # 0 ~ 1 값으로 정규화된 데이터가 입력으로 들어옴
        )

    def forward(self, x):
        feature = self.flow_encoder(x)
        result = self.flow_decoder(feature)
        return result
    
class bbox_ae(nn.Module):
    """Some Information about bbox_ae"""
    def __init__(self, feature):
        super(bbox_ae, self).__init__()

        self.bbox_encoder = nn.Sequential(
            nn.Conv2d(bbox_input_channel, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, feature, 2),
            
        )
        self.bbox_decoder = nn.Sequential(
            nn.ConvTranspose2d(feature, 256, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(4, bbox_input_channel, 3),
            nn.Sigmoid() # 0 ~ 1 값으로 정규화된 데이터가 입력으로 들어옴
        )

    def forward(self, x):
        feature = self.bbox_encoder(x)
        result = self.bbox_decoder(feature)
        return result
    
class ego_ae(nn.Module):
    """Some Information about ego_ae"""
    def __init__(self, feature):
        super(ego_ae, self).__init__()

        self.ego_encoder = nn.Sequential(
            nn.Conv2d(ego_input_channel, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, feature, 2),
            
        )
        self.ego_decoder = nn.Sequential(
            nn.ConvTranspose2d(feature, 256, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(4, ego_input_channel, 3),
            nn.Sigmoid() # 0 ~ 1 값으로 정규화된 데이터가 입력으로 들어옴
        )

    def forward(self, x):
        feature = self.ego_encoder(x)
        result = self.ego_decoder(feature)
        return result
    
class other_SAD(nn.Module):
    """Some Information about other_SAD"""
    def __init__(self, feature):
        super(other_SAD, self).__init__()

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(flow_input_channel, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, feature, 2),
            
        )

        self.bbox_encoder = nn.Sequential(
            nn.Conv2d(bbox_input_channel, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, feature, 2),
            
        )

        self.ego_encoder = nn.Sequential(
            nn.Conv2d(ego_input_channel, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, feature, 2),
            
        )

        self.c = None

    def forward(self, bbox, flow, ego):

        bbox_feature = self.bbox_encoder(bbox)
        flow_feature = self.flow_encoder(flow)
        ego_feature = self.ego_encoder(ego)

        bbox_ratio = (alpha)/(alpha + beta + gamma)
        flow_ratio = (beta)/(alpha + beta + gamma)
        ego_ratio = (gamma)/(alpha + beta + gamma)

        feature = (bbox_feature * bbox_ratio) + (flow_feature * flow_ratio) + (ego_feature * ego_ratio)

        return feature
    
class ego_SAD(nn.Module):
    """Some Information about ego_SAD"""
    def __init__(self, feature):
        super(ego_SAD, self).__init__()

        self.ego_encoder = nn.Sequential(
            nn.Conv2d(ego_input_channel, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, feature, 2),
            
        )

        self.c = None

    def forward(self, ego):

        return self.ego_encoder(ego)
