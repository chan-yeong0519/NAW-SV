import math
import torch
import random
import warnings
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset

from egg_exp import signal_processing, data_augmentation

class NETrainSet(Dataset):
    def __init__(self, vox1, path_musan, path_rir, size):
        self.items = vox1.train_set
        self.musan = data_augmentation.Musan(f'{path_musan}_split/train')
        self.rir = data_augmentation.RIR(path_rir)
        self.crop_size = size

        #####
        # for sampler
        ####
        self.utt_per_spk = {}
        for idx, line in enumerate(vox1.train_set):
            label = vox1.labels[line.path.split("/")[4]]
            if label not in self.utt_per_spk:
                self.utt_per_spk[label] = []
            self.utt_per_spk[label].append(idx)
        print('nb_train_spk: {}'.format(len(self.utt_per_spk)))
        print('nb_train_utt: {}'.format(len(self.items)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, indices):
        # sample
        utts = []
        for i, index in enumerate(indices):
            item = self.items[index]

            # read wav
            wav_clean = signal_processing.rand_crop_read(item.path, self.crop_size)
            if i==0: utts.append(wav_clean)
            else:
                # adding noise
                wav_noise = self.augment(wav_clean)
                utts.append(wav_noise)
                wav_clean = torch.from_numpy(wav_clean)
                wav_ref = wav_clean.clone().detach()
        return utts[0], utts[1], wav_ref, item.label
        
    def augment(self, wav):
        # Musan
        aug_type = random.randint(0, 3)
        if aug_type == 0:
            wav_noise = self.musan(wav, 'noise')
        elif aug_type == 1:
            wav_noise = self.musan(wav, 'speech')
        elif aug_type == 2:
            wav_noise = self.musan(wav, 'music')
        elif aug_type == 3:
            wav_noise = self.musan(wav, 'speech')
            wav_noise = self.musan(wav_noise, 'music')

        return wav_noise
    
    def _make_mask_indices(self, wav):
        mask_indices = []

        while len(mask_indices) < self.num_mask: 
            mask_idx = random.randint(0, wav.shape[0] - self.mask_size - 1)
            if all(abs(mask_idx - i) >= self.mask_size for i in mask_indices):
                mask_indices.append(mask_idx)

        return mask_indices

class Voxceleb_sampler(torch.utils.data.DistributedSampler):
	"""
	Acknowledgement: Github project 'clovaai/voxceleb_trainer'.
	link: https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py
	Adjusted for RawNeXt
	"""
	def __init__(self, dataset, nb_utt_per_spk, max_seg_per_spk, batch_size):
		# distributed settings
		if not torch.distributed.is_available():
			raise RuntimeError("Requires distributed package.")
		self.nb_replicas = torch.distributed.get_world_size()
		self.rank = torch.distributed.get_rank()
		self.epoch = 0

		# sampler config
		self.dataset = dataset
		self.utt_per_spk = dataset.utt_per_spk
		self.nb_utt_per_spk = nb_utt_per_spk
		self.max_seg_per_spk = max_seg_per_spk
		self.batch_size = batch_size
		self.nb_samples = int(
			math.ceil(len(dataset) / self.nb_replicas)
		)  
		self.total_size = (
			self.nb_samples * self.nb_replicas
		) 
		self.__iter__() 

	def __iter__(self):
		
		np.random.seed(self.epoch)

		# speaker ids
		spk_indices = np.random.permutation(list(self.utt_per_spk.keys()))

		# pair utterances by 2
		# list of list
		lol = lambda lst: [lst[i : i + self.nb_utt_per_spk] for i in range(0, len(lst), self.nb_utt_per_spk)]

		flattened_list = []
		flattened_label = []

		# Data for each class
		for findex, key in enumerate(spk_indices):
			# list, utt keys for one speaker
			utt_indices = self.utt_per_spk[key]
			# number of pairs of one speaker's utterances
			nb_seg = round_down(min(len(utt_indices), self.max_seg_per_spk), self.nb_utt_per_spk)
			# shuffle -> make to pairs
			rp = lol(np.random.permutation(len(utt_indices))[:nb_seg])
			flattened_label.extend([findex] * (len(rp)))
			for indices in rp:
				flattened_list.append([utt_indices[i] for i in indices])

		# data in random order
		mixid = np.random.permutation(len(flattened_label))
		mixlabel = []
		mixmap = []

		# prevent two pairs of the same speaker in the same batch
		for ii in mixid:
			startbatch = len(mixlabel) - (
				len(mixlabel) % (self.batch_size * self.nb_replicas)
			)
			if flattened_label[ii] not in mixlabel[startbatch:]:
				mixlabel.append(flattened_label[ii])
				mixmap.append(ii)
		it = [flattened_list[i] for i in mixmap]

		# adjust mini-batch-wise for DDP
		nb_batch, leftover = divmod(len(it), self.nb_replicas * self.batch_size)
		if leftover != 0:
			warnings.warn(
				"leftover:{} in sampler, epoch:{}, gpu:{}, cropping..".format(
					leftover, self.epoch, self.rank
				)
			)
			it = it[: self.nb_replicas * self.batch_size * nb_batch]
		_it = []
		for idx in range(
			self.rank * self.batch_size, len(it), self.nb_replicas * self.batch_size
		):
			_it.extend(it[idx : idx + self.batch_size])
		it = _it
		self._len = len(it)

		return iter(it)

	def __len__(self):
		return self._len


class FTTrainSet(Dataset):
    def __init__(self, items, path_musan, path_rir, size, DA_p=None):
        self.items = items
        self.musan = data_augmentation.Musan(f'{path_musan}_split/train')
        self.rir = data_augmentation.RIR(path_rir)
        self.crop_size = size
        self.DA_p = DA_p

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # sample  
        item = self.items[index]

        # read wav
        wav = signal_processing.rand_crop_read(item.path, self.crop_size)
        
        # for tuning data
        wav = self.augment(wav, self.DA_p)
        
        return wav, item.label
        
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