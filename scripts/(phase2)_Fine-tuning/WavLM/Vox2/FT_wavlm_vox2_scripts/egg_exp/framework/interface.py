import copy
import torch
from typing import Optional
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

class Framework(metaclass=ABCMeta):
    def __init__(self):
        self.modules = {}
        self.freeze_modules = {}
        self.trainable_modules = {}
        self.device = 'cpu'
        self.ddp = False
    
    @abstractmethod
    def __call__(self, x: torch.Tensor, *labels: Optional[torch.Tensor]):
        pass

    def add_module(self, key, model, flag_train=True):
        self.modules[key] = model
        self.set_module_trainability(key, flag_train)
    
    def set_module_trainability(self, key, flag_train):
        for param in self.modules[key].parameters():
            param.requires_grad=flag_train
        if not flag_train:
            self.modules[key].eval()
    
    def use_distributed_data_parallel(self, device, find_unused_parameters=False):
        for key in self.modules.keys():
            self.modules[key].to(device)

            if self.is_trainable(key):
                self.modules[key] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.modules[key])
                self.modules[key] = torch.nn.parallel.DistributedDataParallel(
                        self.modules[key], device_ids=[device], find_unused_parameters=find_unused_parameters)

        self.device = device
        self.ddp = True
        
    def get_parameters(self):
        params = []
        for key, model in self.modules.items():
            if self.is_trainable(key):
                params += list(model.parameters())
        return params
    
    def copy_state_dict(self):
        output = {}
        for key, model in self.modules.items():
            if 0 < len(model.state_dict().keys()):
                output[key] = copy.deepcopy(model.state_dict())
            
        return output
        
    def load_state_dict(self, state_dict):
        for key, params in state_dict.items():
            self.modules[key].load_state_dict(params)

    def eval(self):
        for key in self.modules.keys():
            if self.is_trainable(key):
                self.modules[key].eval()
            
    def train(self):
        for key in self.modules.keys():
            if self.is_trainable(key):
                self.modules[key].train()
            
    def is_trainable(self, key):
        flag_param_exist = 0 < len(self.modules[key].state_dict().keys())
        flag_require_grad = False
        if flag_param_exist:
            for param in self.modules[key].parameters():
                flag_require_grad = param.requires_grad
                if flag_require_grad:
                    break
        return flag_param_exist and flag_require_grad