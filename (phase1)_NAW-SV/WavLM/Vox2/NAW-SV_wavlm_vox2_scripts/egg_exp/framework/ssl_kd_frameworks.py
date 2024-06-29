import torch
from .interface import Framework

class SSL_NE_Framework(Framework):
    KD_MODE = 0
    FT_MODE = 1
    
    def __init__(self, teacher, student, backend, classifier, kd_loss, embed_kd_loss, ft_loss):
        super(SSL_NE_Framework, self).__init__()
        self.add_module('teacher', teacher, flag_train=False)
        self.add_module('student', student, flag_train=True)
        self.add_module('backend', backend, flag_train=True)
        self.add_module('classifier', classifier, flag_train=True)
        self.add_module('kd_loss', kd_loss, flag_train=True)
        self.add_module('embed_kd_loss', embed_kd_loss, flag_train=True)
        self.add_module('ft_loss', ft_loss, flag_train=True)
        
    def __call__(self, x_noisy, x_clean=None, label=None, unfreeze_ssl=True, idx_without_adapter=None, masking=False):
        # student
        if unfreeze_ssl:
            x_noisy = self.modules['student'](x_noisy, output_hidden_states=True, idx_without_adapter=idx_without_adapter, masking=masking)
            x_noisy = torch.stack(x_noisy.hidden_states, dim=1)
        else:
            with torch.set_grad_enabled(False):
                x_noisy = self.modules['student'](x_noisy, output_hidden_states=True, idx_without_adapter=idx_without_adapter, masking=masking)
                x_noisy = torch.stack(x_noisy.hidden_states, dim=1)
        
        kd_loss = None
        ft_loss = None
        if self.mode == self.KD_MODE:
            # loss (KD)
            if x_clean is not None:
                with torch.set_grad_enabled(False):
                    kd_label = self.modules['teacher'](x_clean, output_hidden_states=True).hidden_states
                    kd_label = torch.stack(kd_label, dim=1)
                kd_loss = self.modules['kd_loss'](x_noisy, kd_label)
                x_noisy = self.modules['classifier'](x_noisy)
                kd_label = self.modules['classifier'](kd_label)
                embed_kd_loss = self.modules['embed_kd_loss'](x_noisy, kd_label)
                return kd_loss, embed_kd_loss
            else:
                x_noisy = self.modules['classifier'](x_noisy)
                return x_noisy
        else:
            # loss (FT)
            x_noisy = self.modules['backend'](x_noisy)
            if label is not None:
                ft_loss = self.modules['ft_loss'](x_noisy, label)
                return x_noisy, ft_loss
            else:
                return x_noisy
        
    def set_kd_mode(self, unfreeze_adapter_only):
        self.mode = self.KD_MODE
        self.set_module_trainability('teacher', False)
        self.set_module_trainability('classifier', True)
        self.set_module_trainability('kd_loss', True)
        self.set_module_trainability('embed_kd_loss', True)
        self.set_module_trainability('backend', False)
        self.set_module_trainability('ft_loss', False)
        if unfreeze_adapter_only:
            self.set_module_trainability('student', False)
            for name, param in self.modules['student'].named_parameters():
                if 'adapter' in name:
                    param.requires_grad = True
                    #print(name)
        else:
            self.set_module_trainability('student', True)
            
    def set_ft_mode(self, unfreeze_student):
        self.mode = self.FT_MODE
        self.set_module_trainability('student', unfreeze_student)
        self.set_module_trainability('classifier', False)
        self.set_module_trainability('kd_loss', False)
        self.set_module_trainability('embed_kd_loss', False)
        self.set_module_trainability('backend', True)
        self.set_module_trainability('ft_loss', True)
    
    def set_large_margin(self, cos, sin):
        self.modules['ft_loss'].cos_pos_m = cos
        self.modules['ft_loss'].sin_pos_m = sin
