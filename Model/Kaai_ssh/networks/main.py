
from .GRU_TAD import Encoder_GRU
import networks.seperate_autoencoder_model as sam
import networks.recurrence_autoencoder_model as ram
#from test1 import test1

def build_network(net_name, ae_net=None):
    """Builds the neural network."""
    
    implemented_networks = ('GRU_TAD','Recurrence',)
    
    assert net_name in implemented_networks
    
    net = None
    
    if net_name == 'GRU_TAD':
        net = sam.model()
    elif net_name == 'Recurrence':
        net = ram.recurrence_model()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('GRU_TAD','flow','bbox', 'ego', 'Recurrence')
    
    assert net_name in implemented_networks
    
    ae_net = None
    
    if net_name == 'GRU_TAD':
        return sam.flow_model(), sam.bbox_model(), sam.ego_model()
    elif net_name == 'flow':
        ae_net = sam.flow_model()
    elif net_name == 'bbox':
        ae_net = sam.bbox_model()
    elif net_name == 'ego':
        ae_net = sam.ego_model()
    elif net_name == 'Recurrence':
        return ram.recurrence_flow_autoencoder(), ram.recurrence_bbox_autoencoder(), ram.recurrence_ego_autoencoder()
    return ae_net