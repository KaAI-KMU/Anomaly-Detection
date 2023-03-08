import torch
from torch import nn
from config import model_config as cfg
import numpy as np

class bbox_model(nn.Module):
    """Some Information about model"""
    def __init__(self):
        super(bbox_model, self).__init__()

        self.bbox_encoder = model_bbox_encoder()
        self.bbox_decoder = model_bbox_decoder()

    def forward(self, bbox = None):


        result_bbox = self.bbox_encoder(bbox)
        result_bbox = self.bbox_decoder(result_bbox)
        
        #self.bbox_ratio, self.flow_ratio, self.ego_ratio = (self.bbox_ratio)/(self.bbox_ratio+self.flow_ratio+self.ego_ratio), (self.flow_ratio)/(self.bbox_ratio+self.flow_ratio+self.ego_ratio), (self.ego_ratio)/(self.bbox_ratio+self.flow_ratio+self.ego_ratio)
        return result_bbox
    
class model_bbox_encoder(nn.Module):
    """Some Information about model_bbox_encoder"""
    def __init__(self):
        super(model_bbox_encoder, self).__init__()

        self.layer = nn.Sequential(
            nn.GRUCell(4, cfg.feature_space)
        )

    def forward(self, x, hidden_state = None): # (B, T, 4)
        if hidden_state == None:
            hidden_state = torch.randn(cfg.sequence, cfg.feature_space)
        for index in range(x.shape[1]):
            hidden_state = self.layer(x[:,index, :])
        return hidden_state
    
class model_bbox_decoder(nn.Module):
    """Some Information about model_bbox_encoder"""
    def __init__(self):
        super(model_bbox_decoder, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(cfg.feature_space, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)
    
###############################################################
    
class flow_model(nn.Module):
    """Some Information about model"""
    def __init__(self):
        super(flow_model, self).__init__()
        self.flow_encoder = model_flow_encoder()
        self.flow_decoder = model_flow_decoder()

    def forward(self, flow = None):

        storage = torch.zeros(flow.shape[0], flow.shape[1], flow.shape[-1]).to(flow.device)

        mid_x = flow.shape[2] // 2
        mid_y = flow.shape[3] // 2

        for i in range(flow.shape[0]):
            for j in range(flow.shape[1]):
                for k in range(2):
                    storage[i,j,k] = flow[i,j,mid_x,mid_y,k]

        result_flow = self.flow_encoder(storage)
        result_flow = self.flow_decoder(result_flow)

        return result_flow
        #self.bbox_ratio, self.flow_ratio, self.ego_ratio = (self.bbox_ratio)/(self.bbox_ratio+self.flow_ratio+self.ego_ratio), (self.flow_ratio)/(self.bbox_ratio+self.flow_ratio+self.ego_ratio), (self.ego_ratio)/(self.bbox_ratio+self.flow_ratio+self.ego_ratio)

class model_flow_encoder(nn.Module):
    """Some Information about model_flow_encoder"""
    def __init__(self):
        super(model_flow_encoder, self).__init__()

        self.layer = nn.Sequential(
            nn.GRUCell(2, cfg.feature_space)
        )
    def forward(self, x, hidden_state = None):
        if hidden_state == None:
            hidden_state = torch.randn(cfg.sequence, cfg.feature_space)
        for index in range(x.shape[1]):
            hidden_state = self.layer(x[:,index, :])
        return hidden_state
    
class model_flow_decoder(nn.Module):
    """Some Information about model_flow_encoder"""
    def __init__(self):
        super(model_flow_decoder, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(cfg.feature_space, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
            nn.Linear(2,2)
        )

    def forward(self, x):
        return self.layer(x)
    
##################################################################################

class ego_model(nn.Module):
    """Some Information about model"""
    def __init__(self):
        super(ego_model, self).__init__()

        self.ego_encoder = model_ego_encoder()
        self.ego_decoder = model_ego_decoder()


    def forward(self, ego = None):

        result_ego = self.ego_encoder(ego)
        result_ego = self.ego_decoder(result_ego)
        
        #self.bbox_ratio, self.flow_ratio, self.ego_ratio = (self.bbox_ratio)/(self.bbox_ratio+self.flow_ratio+self.ego_ratio), (self.flow_ratio)/(self.bbox_ratio+self.flow_ratio+self.ego_ratio), (self.ego_ratio)/(self.bbox_ratio+self.flow_ratio+self.ego_ratio)

        return result_ego
    
class model_ego_encoder(nn.Module):
    """Some Information about model_ego_encoder"""
    def __init__(self):
        super(model_ego_encoder, self).__init__()
        self.layer = nn.Sequential(
            nn.GRUCell(3, cfg.feature_space)
        )
    def forward(self, x, hidden_state = None):
        if hidden_state == None:
            hidden_state = torch.randn(cfg.sequence, cfg.feature_space)
        for index in range(x.shape[1]):
            hidden_state = self.layer(x[:,index, :])
        return hidden_state
    
class model_ego_decoder(nn.Module):
    """Some Information about model_ego_encoder"""
    def __init__(self):
        super(model_ego_decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(cfg.feature_space, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Tanh()
        )

        self.pi = np.pi
    def forward(self, x):
        return self.pi * self.layer(x)
    

##########################################################################

class model(nn.Module):
    """Some Information about module"""
    def __init__(self):
        super(model, self).__init__()

        self.bbox_encoder = model_bbox_encoder()
        self.flow_encoder = model_flow_encoder()
        self.ego_encoder = model_ego_encoder()

    def forward(self, bbox = None, flow = None, ego = None):
        mid_x = flow.shape[2] // 2
        mid_y = flow.shape[3] // 2

        bbox_ratio = 0
        flow_ratio = 0
        ego_ratio = 0
    
        result = np.zeros((cfg.batch,cfg.feature_space))
        if flow != None:
            result = result.device(flow.device)
        elif bbox != None:
            result = result.device(bbox.device)
        elif ego != None:
            result = result.device(ego.device)

        if flow != None:
            storage = torch.zeros(flow.shape[0], flow.shape[1], flow.shape[-1]).to(flow.device)

            for i in range(flow.shape[0]):
                for j in range(flow.shape[1]):
                    for k in range(2):
                        storage[i,j,k] = flow[i,j,mid_x,mid_y,k]
            flow_ratio = cfg.beta
            flow_result = self.flow_encoder(storage)
        
        if bbox != None:
            bbox_result = self.bbox_encoder(bbox)
            bbox_ratio = cfg.alpha

        if ego != None:
            ego_result = self.ego_encoder(ego)
            ego_ratio = cfg.gamma

        bbox_ratio, flow_ratio, ego_ratio = (bbox_ratio)/(bbox_ratio+flow_ratio+ego_ratio), (flow_ratio)/(bbox_ratio+flow_ratio+ego_ratio), (ego_ratio)/(bbox_ratio+flow_ratio+ego_ratio)
        result = (flow_result * flow_ratio) + (bbox_result * bbox_ratio) + (ego_result * ego_ratio)

        return result