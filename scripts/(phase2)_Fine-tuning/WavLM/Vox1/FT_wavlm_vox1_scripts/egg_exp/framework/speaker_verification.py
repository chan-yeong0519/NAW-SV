import math
import torch
from .interface import Framework
 
class SSL_Backend_SVFramework(Framework):
    def __init__(self, ssl, backend, criterion):
        super(SSL_Backend_SVFramework, self).__init__()
        self.add_module('ssl', ssl, flag_train=True)
        self.add_module('backend', backend, flag_train=True)
        if criterion is not None:
            self.add_module('criterion', criterion, flag_train=True)
        
    def __call__(self, x, label=None, unfreeze_ssl=False):
        if unfreeze_ssl:
            x = self.modules['ssl'](x, output_hidden_states=True)
            x = torch.stack(x.hidden_states, dim=1)
        else:
            with torch.set_grad_enabled(False):
                x = self.modules['ssl'](x, output_hidden_states=True)
                x = torch.stack(x.hidden_states, dim=1)

        x = self.modules['backend'](x)
        
        if label is None:
            return x
        else:
            loss = self.modules[f'criterion'](x, label)
            return x, loss
        
    def set_ft_mode(self, unfreeze_ssl):
        self.set_module_trainability('ssl', unfreeze_ssl)
        self.set_module_trainability('backend', True)
        self.set_module_trainability('criterion', True)
    
    def set_large_margin(self, cos, sin):
        self.modules['criterion'].cos_pos_m = cos
        self.modules['criterion'].sin_pos_m = sin

    def _load_state_dict(self, ssl_path, backend_path=None, criterion_path=None):
        self.modules['ssl'].load_state_dict(torch.load(ssl_path), strict=True)
        if backend_path is not None:
            self.modules['backend'].load_state_dict(torch.load(backend_path), strict=False)
        if criterion_path is not None:
            self.modules['criterion'].load_state_dict(torch.load(criterion_path), strict=False)