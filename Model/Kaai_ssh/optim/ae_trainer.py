import torch
from torch.nn import functional as F
from tqdm import tqdm
import logging
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import roc_auc_score


def trainer(args, model, train_gen, verbose = True, device = 'cuda', n_epochs=150):
    
    logger = logging.getLogger()

    loader = tqdm(train_gen, total = len(train_gen))

    criterion = nn.MSELoss(reduction='none')
    
    ae_net = model.to(device)
    criterion = criterion.to(device)

    optimizer = optim.Adam(ae_net.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1)
    
    logger.info('Starting pretraining...')
    start_time = time.time()
    ae_net.train()
    
    for epoch in range(n_epochs):
        #optimizer.step()
        scheduler.step()
            
        epoch_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for data in loader:
            bbox_in, flow_in, _, bbox_out, _ = data
            
            optimizer.zero_grad() #얘 위치 어디??
                
            prediction = model(flow_in, bbox_in)
            
            loss = criterion(prediction, bbox_out)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
        epoch_train_time = time.time() - epoch_start_time
        logger.info(f'| Epoch: {epoch + 1:03}/{n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')
        
    train_time = time.time() - start_time
    logger.info('Pretraining Time: {:.3f}s'.format(train_time))
    logger.info('Finished pretraining.')
    
    return ae_net

def validator(n_epochs, args, model, val_gen, verbose = True, device = 'cuda'):
    
    logger = logging.getLogger()

    # Get test data loader
    loader = tqdm(val_gen, total = len(val_gen))
    
    #Set loss
    criterion = nn.MSELoss(reduction='none')
    
    # Set device for network
    ae_net = model.to(device)
    criterion = criterion.to(device)
    
    # Testing
    logger.info('Testing autoencoder...')
    epoch_loss = 0.0
    n_batches = 0
    start_time = time.time()
    idx_label_score = []
    ae_net.eval()
    
    with torch.no_grad():
        for data in loader:
            bbox_in, flow_in, _, bbox_out, _ = data
            
            prediction = model(flow_in, bbox_in)
            loss = criterion(prediction, bbox_out)
            loss = torch.mean(loss)
            epoch_loss += loss.item()
            n_batches += 1
            
    test_time = time.time() - start_time
    
    # Compute AUC
    _, labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)

    # Log results
    logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
    logger.info('Test AUC: {:.2f}%'.format(100. * test_auc))
    logger.info('Test Time: {:.3f}s'.format(test_time))
    logger.info('Finished testing autoencoder.')