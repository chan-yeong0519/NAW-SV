import os
import math
import datetime
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, WavLMModel

try: 
    import egg_exp
except:
    import sys
    sys.path.append('/exp_lib')
from egg_exp import log, dataset, loss, model, framework, util
import data_processing
import preprocessing
import arguments
import train

def run(process_id, args, experiment_args):
    # set reproducible
    torch.cuda.empty_cache()
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # DDP env
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args['port']
    args['rank'] = process_id
    args['device'] = f'cuda:{process_id}'
    torch.distributed.init_process_group(
            backend='nccl', world_size=args['world_size'], rank=args['rank'])
           
    # logger
    if process_id == 0:
        builder = log.LoggerList.Builder(args['name'], args['project'], args['tags'], args['description'], args['path_scripts'], args)
        builder.use_local_logger(args['path_log'])
        #builder.use_neptune_logger(args['neptune_user'], args['neptune_token'])
        #builder.use_wandb_logger(args['wandb_entity'], args['wandb_api_key'], args['wandb_group'])
        logger = builder.build()
        logger.log_arguments(experiment_args)
    else:
        logger = None

    # data loader
    vox2 = dataset.VoxCeleb2(args['path_train'], args['path_test'], args['path_trials'])

    NE_train_set = data_processing.NETrainSet(vox2, args['path_musan'], args['path_rir'], args['crop_size'])
    NE_train_sampler = data_processing.Voxceleb_sampler(dataset=NE_train_set, nb_utt_per_spk=args['nb_utt_per_spk'], max_seg_per_spk=args['max_seg_per_spk'], batch_size=args['kd_batch_size'])
    NE_train_loader = DataLoader(
        NE_train_set,
        num_workers=args['num_workers'],
        batch_size=args['kd_batch_size'],
        pin_memory=True,
        sampler=NE_train_sampler,
        drop_last=True
    )

    enrollment_set_TTA = data_processing.EnrollmentSet(vox2.enrollment_set, args['num_seg'], args['seg_size'])
    enrollment_sampler_TTA = DistributedSampler(enrollment_set_TTA, shuffle=False)
    enrollment_loader_TTA = DataLoader(
        enrollment_set_TTA,
        num_workers=args['num_workers'] * 2,
        batch_size=1,
        sampler=enrollment_sampler_TTA,
        pin_memory=True,
    )

    # criterion
    kd_loss = loss.NE_KDLoss(args['hint_lambda'])
    embed_kd_loss = loss.E_APN(args['device'])
    ft_loss = loss.AAMSoftmax(
        args['embed_size'], 
        len(vox2.class_weight), 
        args['aam_m'], 
        args['aam_s'], 
        class_weight=vox2.class_weight, 
        topk_panelty=args['topk_panelty'])
    
    # teacher network
    config = AutoConfig.from_pretrained(args['huggingface_url'])
    config.mask_time_prob = 0.0
    ssl_clean = WavLMModel.from_pretrained(
        args['huggingface_url'],
        from_tf=bool(".ckpt" in args['huggingface_url']),
        config=config,
        revision="main",
        ignore_mismatched_sizes=False,
    )

    # student network
    ssl_noise = model.Custom_WavLMPlus(
        use_cls_token=False, 
        adapter_hidden_size=args['adapter_hidden_size'],
        mask_time_length=args['mask_time_length'],
        mask_time_prob=args['mask_time_prob'],
        mask_feature_length=args['mask_feature_length'],
        mask_feature_prob=args['mask_feature_prob'],
        )
    
    # classifier
    classifier = model.LinearClassifier(
        args['ssl_hidden_layer_num'], 
        args['ssl_hidden_layer_size'], 
        args['embed_size'], 
        weighted_sum=args['weighted_sum']
    )

     # backend model
    backend = model.SSL_ECAPA_TDNN_small(
        args['ssl_hidden_layer_num'], 
        args['ssl_hidden_layer_size'], 
        args['embed_size'], 
        weighted_sum=args['weighted_sum']
    )

    # set framework
    ssl_NE_framework = framework.SSL_NE_Framework(
        teacher=ssl_clean,
        student=ssl_noise,
        backend=backend,
        classifier=classifier,
        kd_loss=kd_loss,
        embed_kd_loss=embed_kd_loss,
        ft_loss=ft_loss,
    )
    ssl_NE_framework.use_distributed_data_parallel(f'cuda:{process_id}', True)

    #=============================================
    #                   Train 
    #=============================================

    # mode change
    ssl_NE_framework.set_kd_mode(unfreeze_adapter_only=True)

    # optimizer
    optimizer = torch.optim.Adam(ssl_NE_framework.get_parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args['kd_epoch'],
        T_mult=args['T_mult'],
        eta_min=args['lr_min']
    )
    
    # run
    best_eer = 100
    unfreeze_ssl = 10
    best_state = ssl_NE_framework.copy_state_dict()
    for kd_epoch in range(1, args['kd_epoch'] + 1):
        scheduler.step(kd_epoch)
        NE_train_sampler.set_epoch(kd_epoch)
        if kd_epoch == unfreeze_ssl:
            ssl_NE_framework.set_kd_mode(unfreeze_adapter_only=False)
            optimizer = torch.optim.Adam(ssl_NE_framework.get_parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

        # train
        train.train_NE(kd_epoch, ssl_NE_framework, optimizer, NE_train_loader, logger, args['embed_loss_weight'], separate_batch=args['separate_batch'], masking=args['masking'])

        if int(kd_epoch % 10) == 0 and logger is not None:
            best_state = ssl_NE_framework.copy_state_dict()
            for key, v in best_state.items():
                if key == 'student':
                    logger.save_model(f'NAWSV_vox2_wavlm_params_pretrained_model_{kd_epoch}', v)
                elif key == 'classifier':
                    logger.save_model(f'NAWSV_vox2_wavlm_params_{key}_{kd_epoch}', v)

        # test
        enrollment_set_TTA.Key = 'clean'
        eer = train.val(ssl_NE_framework, enrollment_loader_TTA, vox2.trials)
    
        if logger is not None:
            logger.log_metric(f"EER_clean/NAWSV_eer", eer, kd_epoch)
            print('EER_clean/NAWSV_eer: ', eer)
            best_state = ssl_NE_framework.copy_state_dict()

            if best_eer > eer:
                if logger is not None:
                    logger.log_metric('BestEER/NAWSV_eer', eer, kd_epoch)
                best_eer = eer


if __name__ == '__main__':
    # get arguments
    args, system_args, experiment_args = arguments.get_args()
    
    # check gpu environment
    if args['usable_gpu'] is None: 
        args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['usable_gpu']
        args['gpu_ids'] = args['usable_gpu'].split(',')
    assert 0 < len(args['gpu_ids']), 'Only GPU env are supported'
    
    args['port'] = f'10{datetime.datetime.now().microsecond % 100}'

    # set DDP
    args['world_size'] = len(args['gpu_ids'])
    args['kd_batch_size'] = args['kd_batch_size'] // args['world_size']
    if args['kd_batch_size'] % args['world_size'] != 0:
        print(f'The batch size is resized to {args["kd_batch_size"] * args["world_size"]} because the rest are discarded.')
    
    # check dataset
    data_preprocessor = preprocessing.DataPreprocessor(args['path_musan'], args['path_test'])
    data_preprocessor.check_environment()

    # start
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        run,
        nprocs=args['world_size'],
        args=(args, experiment_args)
    )