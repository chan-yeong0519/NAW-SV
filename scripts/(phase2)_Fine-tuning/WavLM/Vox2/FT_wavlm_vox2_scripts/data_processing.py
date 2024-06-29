import random
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset

from egg_exp import signal_processing, data_augmentation

class TrainSet(Dataset):
    def __init__(self, items_ft, path_musan, path_rir, size, DA_p, Train_DA=True):
        self.items_ft = items_ft
        self.musan = data_augmentation.Musan(f'{path_musan}_split/train')
        self.rir = data_augmentation.RIR(path_rir)
        self.crop_size = size
        self.DA_p = DA_p
        self.Train_DA = Train_DA

    def __len__(self):
        return len(self.items_ft)

    def __getitem__(self, index):
        # sample  
        item_ft = self.items_ft[index]
        
        # read wav
        wav_ft = signal_processing.rand_crop_read(item_ft.path, self.crop_size)
        
        # DA
        wav_ft = self.augment(wav_ft, self.DA_p)
        
        return wav_ft, item_ft.label

    def shuffle_kd_items(self):
        random.shuffle(self.items_kd)
        
    def augment(self, wav, DA_p):
        a = random.random()
        if a > DA_p:
            return wav
        else:
            # Musan
            aug_type = random.randint(0, 4)
            if aug_type == 0:
                wav = self.musan(wav, 'noise')
            elif aug_type == 1:
                wav = self.musan(wav, 'speech')
            elif aug_type == 2:
                wav = self.musan(wav, 'music')
            elif aug_type == 3:
                wav = self.musan(wav, 'speech')
                wav = self.musan(wav, 'music')
            elif aug_type == 4:
                wav = self.rir(wav)
                
            return wav
        
class EnrollmentSet(Dataset):
    @property
    def Key(self):
        return self.key
    @Key.setter
    def Key(self, value):
        self.key = value

    def __init__(self, items, num_seg, seg_size):
        self.key = 'clean'
        self.items = items
        self.num_seg = num_seg
        self.seg_size = seg_size
    
    def __len__(self):
        return len(self.items[self.Key])

    def __getitem__(self, index):        
        # sample
        item = self.items[self.Key][index]

        # read wav
        wav_TTA = signal_processing.linspace_crop_read(item.path, self.num_seg, self.seg_size)
        
        return wav_TTA, item.key
	
def round_down(num, divisor):
	return num - (num % divisor)
    
class FullEnrollmentSet(Dataset):
    @property
    def Key(self):
        return self.key
    @Key.setter
    def Key(self, value):
        self.key = value

    def __init__(self, items):
        self.key = 'clean'
        self.items = items
    
    def __len__(self):
        return len(self.items[self.Key])

    def __getitem__(self, index):        
        # sample
        item = self.items[self.Key][index]

        # read wav
        wav, _ = sf.read(item.path)
        
        return wav, item.key