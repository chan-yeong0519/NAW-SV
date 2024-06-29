import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .interface import Criterion

class LastHiddenMSE(Criterion):
    '''KD loss simply comparing the last outputs of teachers and students using MSE. 
    '''
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, x, label):
        return self.mse(x[:, -1, :, :], label[:, -1, :, :])

class NE_KDLoss(Criterion):
    '''SSL KD loss function used in papaer 'Fithubert: Speech representation learning by layer-wise distillation of hidden-unit bert'.
    '''
    def __init__(self, seq_size, ssl_hidden_size, hint_lambda, cos_lambda):
        super(NE_KDLoss, self).__init__()
        self.hint_lambda = hint_lambda
        self.cos_lambda = cos_lambda
        self.log_sigmoid = nn.LogSigmoid()
        #self.cos_sim = nn.CosineSimilarity(dim=1)

        #self.noisy_w = nn.Parameter(torch.FloatTensor(1, seq_size, ssl_hidden_size), requires_grad=True)
        
    def forward(self, x, label):
        # remove CNN output
        x = x[:, 1:, :, :]
        label = label[:, 1:, :, :]
        
        loss_buffer = []
        batch, layer, seq, hidden = x.size()
        for i in range(layer):
            # sample layer
            s_l = x[:, i, :, :].view(batch, -1)
            t_l = label[:, i, :, :].view(batch, -1)
            
            # calculate l2-loss
            l2_loss = torch.mean((s_l - t_l) ** 2, dim=-1)

            # multiply lambda to hint_loss
            l2_loss = l2_loss * self.hint_lambda if i != layer - 1 else l2_loss
            
            loss_buffer.append(l2_loss)
        
        l2_loss_sum = torch.mean(sum(loss_buffer))

        loss = l2_loss_sum

        return loss

class E_APN(Criterion):
    def __init__(self, device, init_w=10.0, init_b=-5.0, **kwargs):
        super(Criterion, self).__init__()

        self.device = device
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.w.requires_grad = True
        self.b.requires_grad = True
        self.cce = nn.CrossEntropyLoss()

        print('Initialised Extended AngleProto')

    def forward(self, x, label=None):
        stepsize = x.size()[0] // 2
        out_clean, out_noisy = x[:stepsize, :], x[stepsize:, :]
        if label is not None:
            label_clean, label2_clean = label[:stepsize, :], label[stepsize:, :]

        cos_sim_matrix_s = F.cosine_similarity(out_noisy.unsqueeze(-1), out_clean.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix_s = cos_sim_matrix_s * self.w + self.b
        if label is not None:
            cos_sim_matrix_s2 = F.cosine_similarity(out_noisy.unsqueeze(-1), label_clean.unsqueeze(-1).transpose(0,2))
            cos_sim_matrix_d = F.cosine_similarity(out_noisy.unsqueeze(-1), label2_clean.unsqueeze(-1).transpose(0,2))
            cos_sim_matrix_s2 = cos_sim_matrix_s2 * self.w + self.b
            cos_sim_matrix_d = cos_sim_matrix_d * self.w + self.b
            cos_sim_matrix = torch.concat((cos_sim_matrix_s, cos_sim_matrix_s2, cos_sim_matrix_d), dim=0)
        else:
            cos_sim_matrix = cos_sim_matrix_s
        
        label = torch.from_numpy(np.asarray(range(0,stepsize))).cuda(self.device)
        if label is not None:
            label = torch.concat((label, label, label))

        criterion = self.cce
        loss = criterion(cos_sim_matrix, label)

        return loss
