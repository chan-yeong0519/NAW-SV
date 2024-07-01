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
* **Authors** :  Chan-yeong Lim*, Hyun-seo Shin*, Ju-ho Kim, Jungwoo Heo, Kyo-Won Koo, Seung-bin Kim, and Ha-Jin Yu (*: Equal contribution)

### Paper abstract
<img src="https://github.com/chan-yeong0519/NAW-SV/blob/main/NAW-SV_framework.PNG" width="600" height="600">
Adopting self-supervised pre-trained models (PMs) in speaker verification (SV) has shown remarkable performance, but their noise robustness is largely unexplored. In the field of automatic speech recognition, additional training strategies enhance the robustness of the models before fine-tuning to improve performance in noisy environments. However, directly applying these strategies to SV risks distorting speaker information. We propose a noise adaptive warm-up training for speaker verification (NAW-SV). The NAW-SV guides the PM to extract consistent representations in noisy conditions using teacher-student learning. In this approach, to prevent the speaker information distortion problem, we introduce a novel loss function called extended angular prototypical network loss, which assists in considering speaker information and exploring robust speaker embedding space. We validated our proposed framework on the noise-synthesized VoxCeleb1 test set, demonstrating promising robustness.

# 1. Prerequisites
## 1.1. Environment Setting

* We used 'nvcr.io/nvidia/pytorch:22.08-py3' image of Nvidia GPU Cloud for conducting our experiments
* The details of the environment settings can be found in 'Dockerfile' file.
* Run 'build.sh' file to make docker image
```
./docker/build.sh
```
(We conducted experiment using 2 or 4 NVIDIA RTX A5000 GPUs)

## 1.2. Datasets
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

1. WavLM-Base+: EER 1.45% and 2.96% in VoxCeleb1, under clean and noisy conditions, respectively. <a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/(phase1)_NAW-SV/WavLM/Vox1/">Link</a>
2. WavLM-Base+: EER 0.85% and 2.31% in VoxCeleb2, under clean and noisy conditions, respectively. <a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/(phase1)_NAW-SV/WavLM/Vox2/">Link</a>

### 2.2. Fine-tuning (phase2)
After the NAW-SV phase, download the weights of HuBERT or WavLM. And then change the weights with the parameters in 'parameters' folder in each experiment folder.

1. WavLM-Base+ (VoxCeleb1). <a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/(phase2)_Fine-tuning/WavLM/Vox1/">Link</a>
2. WavLM-Base+ (VoxCeleb2). <a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/scripts/(phase2)_Fine-tuning/WavLM/Vox2/">Link</a>

(if what you wanted is just testing the model, download the pre-trained model's parameters in the exp_logs. 
Vox1: <a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/exp_logs/(phase1)_NAW-SV/WavLM/Vox1/NAW-SV_wavlm_vox1/models/">Link</a>
, Vox2: <a href="https://github.com/chan-yeong0519/NAW-SV/tree/main/exp_logs/(phase1)_NAW-SV/WavLM/Vox2/NAW-SV_wavlm_vox1/models/">Link</a>
)

And, revise the path of the parameters in main.py. 

```python
# model load
ssl_path = args['path_scripts']+ f'/parameters/NAWSV_vox1_wavlm_params_pretrained_model.pt'
ssl_sv_framework._load_state_dict(ssl_path)
```

### Logger

We have a basic logger that stores information in local. However, if you would like to use an additional online logger (wandb or neptune):

1. In `arguments.py`

```python
# Wandb: Add 'wandb_group', 'wandb_entity' and 'wandb_api_key'
# Neptune: Add 'neptune_user' and 'neptune_token'
# input this arguments in "system_args" dictionary:
# for example
'wandb_group'  : 'exp-group',
'wandb_entity'  : 'user-name',
'wandb_api_key'  : 'WANDB_API_KEY',

'neptune_user'  : 'user-name',
'neptune_token' : 'NEPTUNE_TOKEN'
```

2. In `main.py`

```python
# Just remove "#" in logger which you use
if process_id == 0:
	builder = log.LoggerList.Builder(args['name'], args['project'], args['tags'], args['description'], args['path_scripts'], args)
	builder.use_local_logger(args['path_log'])
	#builder.use_neptune_logger(args['neptune_user'], args['neptune_token'])
	#builder.use_wandb_logger(args['wandb_entity'], args['wandb_api_key'], args['wandb_group'])
	logger = builder.build()
	logger.log_arguments(experiment_args)
else:
	logger = None
```
### 2-3. Run

Just run main.py in scripts!

```python
> python main.py
```


# Citation

Please cite this paper if you make use of the code. 
'''
Will be added after proceeding.
'''
