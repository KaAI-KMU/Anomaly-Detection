import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.DeepSAD_trainer_ego import DeepSADTrainer_ego
from optim.ae_trainer_copy import AETrainer
from networks.seperate_autoencoder_model import bbox_model, flow_model, ego_model, model


class DeepSAD(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.c = None  # hypersphere center c
        self.c_ego = None

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.ae_flow = None
        self.ae_bbox = None
        self.ae_ego = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
            'test_ego_auc': None,
            'test_ego_scores': None
            
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SAD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        self.ego_trainer = DeepSADTrainer_ego(self.c_ego, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.ego_net = self.ego_trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time + self.ego_trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list
        self.c_ego = self.ego_trainer.c.cpu().data.numpy().tolist()

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)
        if self.ego_trainer is None:
            self.ego_trainer = DeepSADTrainer_ego(self.c_ego, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)
        self.ego_trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_ego_auc'] = self.ego_trainer.test_auc
        self.results['test_time'] = self.trainer.test_time + self.ego_trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        self.results['test_ego_scores'] = self.ego_trainer.test_scores

    def pretrain(self, dataset):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_bbox = build_autoencoder('bbox')
        self.ae_flow = build_autoencoder('flow')
        self.ae_ego = build_autoencoder('ego')

        # Train
        self.ae_trainer = AETrainer() #Reconstruct Trainer
        
        self.ae_bbox, self.ae_flow, self.ae_ego = self.ae_trainer.train(dataset, self.ae_bbox, self.ae_flow, self.ae_ego)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        #self.ae_trainer.test(dataset, self.ae_net)

        # Get test results
        #self.ae_results['test_auc'] = self.ae_trainer.test_auc
        #self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        

        flow_dict = self.ae_flow.state_dict()
        bbox_dict = self.ae_bbox.state_dict()
        ego_dict = self.ae_ego.state_dict()

        # Filter out decoder network keys
        flow_dict = {k: v for k, v in flow_dict.items() if k in net_dict}
        bbox_dict = {k: v for k, v in bbox_dict.items() if k in net_dict}
        ego_dict = {k: v for k, v in ego_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(flow_dict)
        net_dict.update(bbox_dict)
        net_dict.update(ego_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()
        flow_dict = self.ae_flow.state_dict() if save_ae else None
        bbox_dict = self.ae_bbox.state_dict() if save_ae else None
        ego_dict = self.ae_ego.state_dict() if save_ae else None

        torch.save({'c': self.c,
                    'c_ego': self.c_ego,
                    'net_dict': net_dict,
                    'flow_dict': flow_dict,
                    'bbox_dict': bbox_dict,
                    'ego_dict' : ego_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.c_ego = model_dict['c_ego']
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_flow is None:
                self.ae_flow = build_autoencoder('flow')
            self.ae_flow.load_state_dict(model_dict['flow_dict'])
            if self.ae_bbox is None:
                self.ae_bbox = build_autoencoder('bbox')
            self.ae_flow.load_state_dict(model_dict['bbox_dict'])
            if self.ae_ego is None:
                self.ae_ego = build_autoencoder('ego')
            self.ae_ego.load_state_dict(model_dict['ego_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)