import os

from ._dataclass import SV_Trial, SV_TrainItem, SV_EnrollmentItem

class VoxCeleb1:
    NUM_TRAIN_ITEM = 148642
    NUM_TRAIN_SPK = 1211
    NUM_TRIALS = 37611

    def __init__(self, path_train, path_test, path_trials):
        self.train_set = []
        self.trials = []
        self.class_weight = []

        # train_set
        labels = {}
        num_utt = [0 for _ in range(self.NUM_TRAIN_SPK)]
        num_sample = 0
        for root, _, files in os.walk(path_train):
            for file in files:
                if '.wav' in file:
                    # combine path
                    f = os.path.join(root, file)
                    
                    # parse speaker
                    spk = f.split('/')[-3]
                    
                    # labeling
                    try: labels[spk]
                    except: 
                        labels[spk] = len(labels.keys())

                    # init item
                    item = SV_TrainItem(path=f, speaker=spk, label=labels[spk])
                    self.train_set.append(item)
                    num_sample += 1
                    num_utt[labels[spk]] += 1

        for n in num_utt:
            self.class_weight.append(num_sample / n)
                    
        '''# test_set
        for root, _, files in os.walk(os.path.join(path_test, 'test')):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = SV_EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.enrollment_set.append(item)'''

        # enrollment_set
        path_test_noise = os.path.join(path_test, 'test_noise')
        path_test = os.path.join(path_test,'test')
        self.enrollment_set = {
            'clean': [],
            'noise_0': [],
            'noise_5': [],
            'noise_10': [],
            'noise_15': [],
            'noise_20': [],
            'speech_0': [],
            'speech_5': [],
            'speech_10': [],
            'speech_15': [],
            'speech_20': [],
            'music_0': [],
            'music_5': [],
            'music_10': [],
            'music_15': [],
            'music_20': []
        }
        self._parse_enrollment(path_test, 'clean')
        self._parse_enrollment(f'{path_test_noise}/noise_0', 'noise_0')
        self._parse_enrollment(f'{path_test_noise}/noise_5', 'noise_5')
        self._parse_enrollment(f'{path_test_noise}/noise_10', 'noise_10')
        self._parse_enrollment(f'{path_test_noise}/noise_15', 'noise_15')
        self._parse_enrollment(f'{path_test_noise}/noise_20', 'noise_20')

        self._parse_enrollment(f'{path_test_noise}/speech_0', 'speech_0')
        self._parse_enrollment(f'{path_test_noise}/speech_5', 'speech_5')
        self._parse_enrollment(f'{path_test_noise}/speech_10', 'speech_10')
        self._parse_enrollment(f'{path_test_noise}/speech_15', 'speech_15')
        self._parse_enrollment(f'{path_test_noise}/speech_20', 'speech_20')

        self._parse_enrollment(f'{path_test_noise}/music_0', 'music_0')
        self._parse_enrollment(f'{path_test_noise}/music_5', 'music_5')
        self._parse_enrollment(f'{path_test_noise}/music_10', 'music_10')
        self._parse_enrollment(f'{path_test_noise}/music_15', 'music_15')
        self._parse_enrollment(f'{path_test_noise}/music_20', 'music_20')

        

        self.trials = self.parse_trials(os.path.join(path_trials, 'trials.txt'))

        # error check
        assert len(self.train_set) == self.NUM_TRAIN_ITEM
        assert len(self.trials) == self.NUM_TRIALS
        assert len(labels) == self.NUM_TRAIN_SPK
        assert len(self.enrollment_set["clean"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["clean"])}'
        assert len(self.enrollment_set["noise_0"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_0"])}'
        assert len(self.enrollment_set["noise_5"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_5"])}'
        assert len(self.enrollment_set["noise_10"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_10"])}'
        assert len(self.enrollment_set["noise_15"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_15"])}'
        assert len(self.enrollment_set["noise_20"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_20"])}'
        assert len(self.enrollment_set["noise_0"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_0"])}'
        assert len(self.enrollment_set["noise_5"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_5"])}'
        assert len(self.enrollment_set["noise_10"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_10"])}'
        assert len(self.enrollment_set["noise_15"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_15"])}'
        assert len(self.enrollment_set["noise_20"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_20"])}'
        assert len(self.enrollment_set["noise_0"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_0"])}'
        assert len(self.enrollment_set["noise_5"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_5"])}'
        assert len(self.enrollment_set["noise_10"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_10"])}'
        assert len(self.enrollment_set["noise_15"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_15"])}'
        assert len(self.enrollment_set["noise_20"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_20"])}'

    def parse_trials(self, path):
        trials = []
        for line in open(path).readlines():
            strI = line.split(' ')
            item = SV_Trial(
                key1=strI[1].replace('\n', ''), 
                key2=strI[2].replace('\n', ''), 
                label=int(strI[0])
            )
            trials.append(item)
        return trials
    
    def _parse_enrollment(self, path, key):	
            for root, _, files in os.walk(path):
                for file in files:
                    if '.wav' in file:
                        temp = os.path.join(root, file)
                        self.enrollment_set[key].append(
                            SV_EnrollmentItem(
                                path=temp,
                                key='/'.join(temp.split('/')[-3:])
                            )
                        )

class VoxCeleb2:
    NUM_TRAIN_ITEM = 1092009
    NUM_TRAIN_SPK = 5994
    NUM_TRIALS = 37611
    #NUM_TRIALS_E = 579818
    #NUM_TRIALS_H = 550894

    def __init__(self, path_train, path_test, path_trials):
        self.train_set = []
        #self.test_set_O = []
        #self.test_set_E = []
        self.trials = []
        #self.trials_H = []
        #self.trials_E = []
        self.class_weight = []

        # train_set
        self.labels = {}
        num_utt = [0 for _ in range(self.NUM_TRAIN_SPK)]
        num_sample = 0
        for root, _, files in os.walk(path_train):
            for file in files:
                if '.wav' in file:
                    # combine path
                    f = os.path.join(root, file)
                    
                    # parse speaker
                    spk = f.split('/')[-3]
                    
                    # labeling
                    try: self.labels[spk]
                    except: 
                        self.labels[spk] = len(self.labels.keys())

                    # init item
                    item = SV_TrainItem(path=f, speaker=spk, label=self.labels[spk])
                    self.train_set.append(item)
                    num_sample += 1
                    num_utt[self.labels[spk]] += 1

        for n in num_utt:
            self.class_weight.append(num_sample / n)
                    
        '''# test_set_O
        for root, _, files in os.walk(os.path.join(path_test, 'test')):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = SV_EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_set_O.append(item)

        # test_set_E
        for root, _, files in os.walk(os.path.join(path_test)):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = SV_EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_set_E.append(item)'''
        
        # enrollment_set
        path_test_noise = os.path.join(path_test, 'test_noise')
        path_test = os.path.join(path_test,'test')
        self.enrollment_set = {
            'clean': [],
            'noise_0': [],
            'noise_5': [],
            'noise_10': [],
            'noise_15': [],
            'noise_20': [],
            'speech_0': [],
            'speech_5': [],
            'speech_10': [],
            'speech_15': [],
            'speech_20': [],
            'music_0': [],
            'music_5': [],
            'music_10': [],
            'music_15': [],
            'music_20': []
        }
        self._parse_enrollment(path_test, 'clean')
        self._parse_enrollment(f'{path_test_noise}/noise_0', 'noise_0')
        self._parse_enrollment(f'{path_test_noise}/noise_5', 'noise_5')
        self._parse_enrollment(f'{path_test_noise}/noise_10', 'noise_10')
        self._parse_enrollment(f'{path_test_noise}/noise_15', 'noise_15')
        self._parse_enrollment(f'{path_test_noise}/noise_20', 'noise_20')

        self._parse_enrollment(f'{path_test_noise}/speech_0', 'speech_0')
        self._parse_enrollment(f'{path_test_noise}/speech_5', 'speech_5')
        self._parse_enrollment(f'{path_test_noise}/speech_10', 'speech_10')
        self._parse_enrollment(f'{path_test_noise}/speech_15', 'speech_15')
        self._parse_enrollment(f'{path_test_noise}/speech_20', 'speech_20')

        self._parse_enrollment(f'{path_test_noise}/music_0', 'music_0')
        self._parse_enrollment(f'{path_test_noise}/music_5', 'music_5')
        self._parse_enrollment(f'{path_test_noise}/music_10', 'music_10')
        self._parse_enrollment(f'{path_test_noise}/music_15', 'music_15')
        self._parse_enrollment(f'{path_test_noise}/music_20', 'music_20')

        self.trials = self.parse_trials(os.path.join(path_trials, 'trials.txt'))

        # error check
        assert len(self.train_set) == self.NUM_TRAIN_ITEM
        assert len(self.trials) == self.NUM_TRIALS
        assert len(self.labels) == self.NUM_TRAIN_SPK
        assert len(self.enrollment_set["clean"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["clean"])}'
        assert len(self.enrollment_set["noise_0"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_0"])}'
        assert len(self.enrollment_set["noise_5"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_5"])}'
        assert len(self.enrollment_set["noise_10"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_10"])}'
        assert len(self.enrollment_set["noise_15"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_15"])}'
        assert len(self.enrollment_set["noise_20"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_20"])}'
        assert len(self.enrollment_set["noise_0"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_0"])}'
        assert len(self.enrollment_set["noise_5"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_5"])}'
        assert len(self.enrollment_set["noise_10"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_10"])}'
        assert len(self.enrollment_set["noise_15"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_15"])}'
        assert len(self.enrollment_set["noise_20"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_20"])}'
        assert len(self.enrollment_set["noise_0"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_0"])}'
        assert len(self.enrollment_set["noise_5"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_5"])}'
        assert len(self.enrollment_set["noise_10"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_10"])}'
        assert len(self.enrollment_set["noise_15"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_15"])}'
        assert len(self.enrollment_set["noise_20"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_20"])}'

        
        '''self.trials_O = self.parse_trials(os.path.join(path_trials, 'trials.txt'))
        self.trials_E = self.parse_trials(os.path.join(path_trials, 'trials_E.txt'))
        self.trials_H = self.parse_trials(os.path.join(path_trials, 'trials_H.txt'))

        # error check
        assert len(self.train_set) == self.NUM_TRAIN_ITEM
        assert len(self.trials_O) == self.NUM_TRIALS
        assert len(self.trials_E) == self.NUM_TRIALS_E
        assert len(self.trials_H) == self.NUM_TRIALS_H
        assert len(labels) == self.NUM_TRAIN_SPK'''

    def parse_trials(self, path):
        trials = []
        for line in open(path).readlines():
            strI = line.split(' ')
            item = SV_Trial(
                key1=strI[1].replace('\n', ''), 
                key2=strI[2].replace('\n', ''), 
                label=int(strI[0])
            )
            trials.append(item)
        return trials
    
    def _parse_enrollment(self, path, key):	
            for root, _, files in os.walk(path):
                for file in files:
                    if '.wav' in file:
                        temp = os.path.join(root, file)
                        self.enrollment_set[key].append(
                            SV_EnrollmentItem(
                                path=temp,
                                key='/'.join(temp.split('/')[-3:])
                            )
                        )
  