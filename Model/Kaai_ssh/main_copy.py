import torch
import logging
import random
import numpy as np

from utils.config import Config
from DeepSAD import DeepSAD
from datasets.main import load_dataset

from config.main_config import *

################################################################################
# Settings
################################################################################

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + 'log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)
    
    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)

    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Network: %s' % net_name)

    # Print model configuration
    logger.info('Eta-parameter: %.2f' % eta)
    # Set seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % seed)

    # Default device to 'cpu' if cuda is not available
    global device
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(eta)
    deepSAD.set_network(net_name)
    
    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_model)
        
    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % ae_optimizer_name)
        logger.info('Pretraining learning rate: %g' % ae_lr)
        logger.info('Pretraining epochs: %d' % ae_n_epochs)
        logger.info('Pretraining learning rate scheduler milestones: %s' % ae_lr_milestone)
        logger.info('Pretraining batch size: %d' % ae_batch_size)
        logger.info('Pretraining weight decay: %g' % ae_weight_decay)

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain()
        
        deepSAD.save_ae_results(export_json=xp_path + '/ae_results.json')
        
    # Log training details
    logger.info('Training optimizer: %s' % optimizer_name)
    logger.info('Training learning rate: %g' % lr)
    logger.info('Training epochs: %d' % n_epochs)
    logger.info('Training learning rate scheduler milestones: %s' % lr_milestone)
    logger.info('Training batch size: %d' % batch_size)
    logger.info('Training weight decay: %g' % weight_decay)
    
    #anomaly dataset for semi supervised learning

    # Train model on dataset
    deepSAD.train()
        
    # Test model
    deepSAD.test(device=device, n_jobs_dataloader=n_jobs_dataloader)
    
    # Save results, model, and configuration
    deepSAD.save_results(export_json=xp_path + '/results.json')
    deepSAD.save_model(export_model=xp_path + '/model.tar')
    #cfg.save_config(export_json=xp_path + '/config.json')

if __name__ == '__main__':
    main()
