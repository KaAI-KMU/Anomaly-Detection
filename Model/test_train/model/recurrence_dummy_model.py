import torch
import torch.nn as nn
from config.model_config import bbox_input_channel, flow_input_channel, ego_input_channel

class dummy_bbox(nn.Module):
    """Some Information about dummy_bbox"""
    def __init__(self):
        super(dummy_bbox, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(bbox_input_channel, bbox_input_channel, 3, padding = 'same')
        )

    def forward(self, x):
        
        return self.layer(x)
    

class dummy_flow(nn.Module):
    """Some Information about dummy_flow"""
    def __init__(self):
        super(dummy_flow, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(flow_input_channel, flow_input_channel, 3, padding = 'same')
        )
    def forward(self, x):

        return self.layer(x)
    
class dummy_ego(nn.Module):
    """Some Information about dummy_ego"""
    def __init__(self):
        super(dummy_ego, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(ego_input_channel, ego_input_channel, 3, padding = 'same')
        )
    def forward(self, x):

        return self.layer(x)