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
    vox1 = dataset.VoxCeleb1(args['path_train'], args['path_test'], args['path_trials'])
    
    train_set = data_processing.TrainSet(vox1.train_set, args['path_musan'], args['path_rir'], args['crop_size'], args['DA_p'])
    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(
        train_set,
        num_workers=args['num_workers'],
        batch_size=args['batch_size'],
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    enrollment_set_TTA = data_processing.EnrollmentSet(vox1.enrollment_set, args['num_seg'], args['seg_size'])
    enrollment_sampler_TTA = DistributedSampler(enrollment_set_TTA, shuffle=False)
    enrollment_loader_TTA = DataLoader(
        enrollment_set_TTA,
        num_workers=args['num_workers'] * 2,
        batch_size=1,
        sampler=enrollment_sampler_TTA,
        pin_memory=True,
    )

    enrollment_set_full = data_processing.FullEnrollmentSet(vox1.enrollment_set)
    enrollment_sampler_full = DistributedSampler(enrollment_set_full, shuffle=False)
    enrollment_loader_full = DataLoader(
        enrollment_set_full,
        num_workers=args['num_workers'] * 2,
        batch_size=1,
        sampler=enrollment_sampler_full,
        pin_memory=True,
    )
    
    # criterion
    criterion = loss.AAMSoftmax(
        args['embed_size'], 
        len(vox1.class_weight), 
        args['aam_m'], 
        args['aam_s'], 
        class_weight=vox1.class_weight, 
        topk_panelty=args['topk_panelty'])
    
    # model
    classifier = model.SSL_ECAPA_TDNN_small(
        args['ssl_hidden_layer_num'], 
        args['ssl_hidden_layer_size'], 
        args['embed_size'], 
        weighted_sum=args['weighted_sum']
    )
    
    # ssl model
    ssl = model.Custom_WavLMPlus(
        use_cls_token=False, 
        adapter_hidden_size=args['adapter_hidden_size'],
        mask_time_length=args['mask_time_length'],
        mask_time_prob=args['mask_time_prob'],
        mask_feature_length=args['mask_feature_length'],
        mask_feature_prob=args['mask_feature_prob'],
    )

    # set framework
    ssl_sv_framework = framework.SSL_Backend_SVFramework(
        ssl=ssl,
        backend=classifier,
        criterion=criterion,
    )
    ssl_sv_framework.use_distributed_data_parallel(f'cuda:{process_id}', True)

    #=============================================
    #                   Train 
    #=============================================
    # model load
    ssl_path = args['path_scripts']+ f'/parameters/NAWSV_vox1_wavlm_params_pretrained_model.pt'
    ssl_sv_framework._load_state_dict(ssl_path)

    # optimizer
    ssl_sv_framework.set_ft_mode(unfreeze_ssl=True)
    optimizer = torch.optim.Adam(ssl_sv_framework.get_parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args['epoch'],
        T_mult=args['T_mult'],
        eta_min=args['lr_min']
    )

    # mode change
    ssl_sv_framework.set_ft_mode(unfreeze_ssl=False)
    
    # run
    best_eer = 100
    noise_type = ['noise', 'music', 'speech']
    snr_type = ['0', '5', '10', '15', '20']
    best_state = ssl_sv_framework.copy_state_dict()
    for epoch in range(0, args['epoch']):
        scheduler.step(epoch)
        train_sampler.set_epoch(epoch)

        if epoch == args['unfreeze_ssl_epoch']:
            ssl_sv_framework.set_ft_mode(unfreeze_ssl=True)
        elif epoch == args['large_margin_epoch']:
            # large margine fine-tuning strategy
            train_set.crop_size = 600 * 160
            cos = math.cos(0.4)
            sin = math.sin(0.4)
            ssl_sv_framework.set_large_margin(cos, sin)

        # train
        train.train_ft(epoch, ssl_sv_framework, optimizer, train_loader, logger, args['unfreeze_ssl_epoch'])

        # test
        enrollment_set_TTA.Key = 'clean'
        eer = train.val(ssl_sv_framework, enrollment_loader_TTA, vox1.trials)
    
        if logger is not None:
            logger.log_metric(f"EER_clean", eer, epoch)
            print('EER_clean: ', eer)
            best_state = ssl_sv_framework.copy_state_dict()

            if best_eer > eer:
                if logger is not None:
                    logger.log_metric('BestEER', eer, epoch)
                    for key, v in best_state.items():
                        if key == 'ssl':
                            logger.save_model(f'FT_vox1_wavlm_params_pretrained_model_{epoch}', v)
                        else:
                            logger.save_model(f'FT_vox1_wavlm_params_{key}_{epoch}', v)
        
        # evaluation (clean_full & noise_utt)
        if best_eer > eer and epoch > 8:
            enrollment_set_TTA.Key = 'clean'
            eer_full = train.eval(ssl_sv_framework, enrollment_loader_full, enrollment_loader_TTA, vox1.trials)
            print('EER_clean_full: ', eer_full)
            for noise in noise_type:
                for snr in snr_type:
                    set_key = noise + '_' + snr
                    enrollment_set_TTA.Key = set_key
                    print(set_key)
                    noise_eer = train.noise_eer(ssl_sv_framework, enrollment_loader_TTA, vox1.trials)
                    if logger is not None:
                        logger.log_metric(f'EER_{set_key}', noise_eer, epoch)
            if logger is not None:
                logger.log_metric('EER_full', eer_full, epoch)
        
        if best_eer > eer:
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
    args['batch_size'] = args['batch_size'] // args['world_size']
    if args['batch_size'] % args['world_size'] != 0:
        print(f'The batch size is resized to {args["batch_size"] * args["world_size"]} because the rest are discarded.')
    
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