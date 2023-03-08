import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder_GRU(nn.Module):
	
	def __init__(self, args):
		super(Encoder_GRU, self).__init__()
		self.args = args
		self.enc = nn.GRUCell(input_size=4, hidden_size=args.hidden_dim)
		
	def forward(self, x, h_init):
		h = self.enc(x, h_init)
		return h

class Decoder_GRU(nn.Module):
    def __init__(self, args):
        super(Decoder_GRU, self).__init__()
        self.args = args
        # PREDICTOR INPUT FC
        self.hidden_to_pred_input = nn.Sequential(nn.Linear(self.args.dec_hidden_size,
                                                            self.args.predictor_input_size),
                                                  nn.ReLU())
        
        # PREDICTOR DECODER
        self.dec = nn.GRUCell(input_size=self.args.predictor_input_size,
                                        hidden_size=self.args.dec_hidden_size)
        
        # PREDICTOR OUTPUT
        if self.args.non_linear_output:
            self.hidden_to_pred = nn.Sequential(nn.Linear(self.args.dec_hidden_size, 
                                                            self.args.pred_dim),
                                                nn.Tanh())
        else:
            self.hidden_to_pred = nn.Linear(self.args.dec_hidden_size, 
                                                            self.args.pred_dim)
                
    def forward(self, h, embedded_ego_pred=None):
        '''
        A RNN preditive model for future observation prediction
        Params:
            h: hidden state tensor from the encoder, (batch_size, enc_hidden_size)
            embedded_ego_pred: (batch_size, pred_timesteps, input_embed_size)
        '''
        output = torch.zeros(h.shape[0], self.args.pred_timesteps, self.args.pred_dim).to(device)

        all_pred_h = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.dec_hidden_size]).to(device)
        all_pred_inputs = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.predictor_input_size]).to(device)
        
        # initial predict input is zero???
        pred_inputs = torch.zeros(h.shape[0], self.args.predictor_input_size).to(device) #self.hidden_to_pred_input(h)
        for i in range(self.args.pred_timesteps):
            if self.args.with_ego:
                pred_inputs = (embedded_ego_pred[:, i, :] + pred_inputs)/2 # average concat of future ego motion and prediction inputs
            all_pred_inputs[:, i, :] = pred_inputs
            h = self.dec(pred_inputs, h)
            
            pred_inputs = self.hidden_to_pred_input(h)

            all_pred_h[:,i,:] = h

            output[:,i,:] = self.hidden_to_pred(h)

        return output, all_pred_h, all_pred_inputs
    
#class GRU_CBAM()