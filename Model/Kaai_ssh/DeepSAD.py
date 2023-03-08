import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.ae_trainer_copy import AETrainer

from config.main_config import *

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

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_bbox, self.ae_flow, self.ae_ego = None, None, None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
        }

        self.ae_results = {
        }

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self):
        """Trains the Deep SAD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(self.c)
        # Get the model
        self.net = self.trainer.train(self.net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(self.net)

        # Get results
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_flow, self.ae_bbox, self.ae_ego = build_autoencoder(self.net_name)

        # Train
        self.ae_trainer = AETrainer(net_name=self.net_name) #Reconstruct Trainer
        
        self.ae_flow, self.ae_bbox, self.ae_ego = self.ae_trainer.train(self.ae_flow, self.ae_bbox, self.ae_ego)

        # Get train results
        self.ae_trainer.validation(self.ae_flow, self.ae_bbox, self.ae_ego)

        self.ae_results['train_time'] = self.ae_trainer.train_time
        self.ae_results['train_loss'] = self.ae_trainer.train_loss
        self.ae_results['validation_time'] = self.ae_trainer.validation_time
        self.ae_results['validation_loss'] = self.ae_trainer.validartion_loss

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
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)