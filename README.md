The code will be uploaded as soon as possible after cleaning it up.

<h1 align="center">
    <b>NAW-SV</b>
</h1>

<h2 align="center">
    Improving Noise Robustness in Self-supervised Pre-trained Model for Speaker Verification
</h2>

<h3 align="left">
	<p>
	<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
	<a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-08.html#rel-22-08"><img src="https://img.shields.io/badge/22.08-2496ED?style=for-the-badge&logo=Docker&logoColor=white"></a>
	<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">
	<a href="https://huggingface.co/"><img src="https://github.com/chan-yeong0519/NAW-SV/blob/main/icon_hugging_face.png?raw=true"></a>
	</p>
</h3>

This repository offers source code for the following paper:

* **Title** : Improving Noise Robustness in Self-supervised Pre-trained Model for Speaker Verification (Accepted for Interspeech 2024)
* **Authors** :  Chan-yeong Lim, Hyun-seo Shin, Ju-ho Kim, Jungwoo Heo, Kyo-Won Koo, Seung-bin Kim, and Ha-Jin Yu

### Paper abstract
<img src="https://github.com/chan-yeong0519/NAW-SV/blob/main/NAW-SV_framework.PNG">
Adopting self-supervised pre-trained models (PMs) in speaker verification (SV) has shown remarkable performance, but their noise robustness is largely unexplored. In the field of automatic speech recognition, additional training strategies enhance the robustness of the models before fine-tuning to improve performance in noisy environments. However, directly applying these strategies to SV risks distorting speaker information. We propose a noise adaptive warm-up training for speaker verification (NAW-SV). The NAW-SV guides the PM to extract consistent representations in noisy conditions using teacher-student learning. In this approach, to prevent the speaker information distortion problem, we introduce a novel loss function called extended angular prototypical network loss, which assists in considering speaker information and exploring robust speaker embedding space. We validated our proposed framework on the noise-synthesized VoxCeleb1 test set, demonstrating promising robustness.

# Prerequisites
## Environment Setting

* We used 'nvcr.io/nvidia/pytorch:22.08-py3' image of Nvidia GPU Cloud for conducting our experiments
* The details of the environment settings can be found in 'Dockerfile' file.
* Run 'build.sh' file to make docker image
```
./docker/build.sh
```
(We conducted experiment using 2 or 4 NVIDIA RTX A5000 GPUs)

## Datasets
* We used VoxCeleb1 & 2 dataset for training and test
* For evaluating the model in noisy conditions, we utilized MUSAN and Nonspeech100 dataset.
* In the fine-tuning stage, for data augmentation, the MUSAN training subset and RIR reverberation datasets were employed.

## 2. Run experiment
Set experimental arguments in `arguments.py` file. Here is list of system arguments to set.

```python
1. 'usable_gpu': {YOUR_PATH} # ex) '0,1,2,3'
	'path_log' is path of saving experiments.
	input type is str

2. 'path_...': {YOUR_PATH}
	'path_...' is path where ... dataset is stored.
	input type is str
```

&nbsp;
### 2.1. NAW-SV (phase1)
You can get the experimental code via hyperlinks. 
<br> Note that we provide our **trained model weights** and **training logs** (such as loss, validation results) for re-implementation. You can find these in 'exps_logs' folder stored in each experiment folder. 

1. HuBERT-Base: EER 1.89% and 4.09% in VoxCeleb1, under clean and noisy conditions, respectively. (<a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/HuBERT_Vox1/NAW-SV(phase1)">
2. HuBERT-Base: EER 1.12% and 2.89% in VoxCeleb2, under clean and noisy conditions, respectively. (<a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/HuBERT_Vox2/NAW-SV(phase1)">
3. WavLM-Base+: EER 1.45% and 2.96% in VoxCeleb1, under clean and noisy conditions, respectively. (<a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/WavLM_Vox1/NAW-SV(phase1)">
4. WavLM-Base+: EER 0.85% and 2.31% in VoxCeleb2, under clean and noisy conditions, respectively. (<a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/WavLM_Vox2/NAW-SV(phase1)">

### 2.2. Fine-tuning (phase2)
After the NAW-SV phase, download the weights of HuBERT or WavLM. And then change the weights with the parameters in 'params' folder in each experiment folder.
(if what you wanted is just testing the model, don't change the weights.)

1. HuBERT-Base (VoxCeleb1). (<a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/HuBERT_Vox1/Fine-tuning(phase2)">
2. HuBERT-Base (VoxCeleb2). (<a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/HuBERT_Vox2/Fine-tuning(phase2)">
3. WavLM-Base+ (VoxCeleb1). (<a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/WavLM_Vox1/Fine-tuning(phase2)">
4. WavLM-Base+ (VoxCeleb2). (<a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/WavLM_Vox2/Fine-tuning(phase2)">

### Logger

We have a basic logger that stores information in local. However, if you would like to use an additional online logger (wandb or neptune):

1. In `arguments.py`

```python
# Wandb: Add 'wandb_user' and 'wandb_token'
# Neptune: Add 'neptune_user' and 'neptune_token'
# input this arguments in "system_args" dictionary:
# for example
'wandb_user'   : 'user-name',
'wandb_token'  : 'WANDB_TOKEN',

'neptune_user'  : 'user-name',
'neptune_token' : 'NEPTUNE_TOKEN'
```

2. In `main.py`

```python
# Just remove "#" in logger which you use

logger = LogModuleController.Builder(args['name'], args['project'],
        ).tags(args['tags']
        ).description(args['description']
        ).save_source_files(args['path_scripts']
        ).use_local(args['path_log']
        #).use_wandb(args['wandb_user'], args['wandb_token'] <- here
        #).use_neptune(args['neptune_user'], args['neptune_token'] <- here
        ).build()
```
### 2-3. Run!

Just run main.py in scripts!

```python
> python main.py
```


# Citation

Please cite this paper if you make use of the code. 
'''
Will be added after the proceeding.
'''
