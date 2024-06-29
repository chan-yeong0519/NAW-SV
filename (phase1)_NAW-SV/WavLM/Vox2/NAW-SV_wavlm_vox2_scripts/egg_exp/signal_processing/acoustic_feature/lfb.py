import torch
import torch.nn as nn

from . import _processing

class LFB(nn.Module):
    def __init__(self, sample_rate=16000, n_lbf=60, coef=0.97, 
                        n_fft=512, win_length=320, hop=160, 
                        with_delta=False, with_emphasis=True, with_energy=False, 
                        frq_mask=False, p=0, max=0):
        super(LFB, self).__init__()
        self.frontend = _processing.LFB(
            win_length,
            hop,
            n_fft,
            sample_rate,
            n_lbf,
            with_energy=with_energy,
            with_emphasis=with_emphasis,
            with_delta=with_delta)
        self.device = 'cpu'
        self.frq_mask = frq_mask
        if frq_mask:
            self.frq_masking = _processing.FrequencyMasking(p, max)

    def forward(self, x):
        assert len(x.size()) == 2, f'Input size error in LFB. Need 2, but get {len(x.size())}'
        
        # device sync
        if x.device != self.device:
            self.device = x.device
            self.to(x.device)
        
        with torch.no_grad():
            x = self.frontend(x)
            if self.frq_mask:
                x = self.frq_masking(x)
            
            return x