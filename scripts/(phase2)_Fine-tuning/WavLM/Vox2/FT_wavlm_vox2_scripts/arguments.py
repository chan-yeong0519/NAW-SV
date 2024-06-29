import os
import itertools

def get_args():
    """
    Returns
        system_args (dict): path, log setting
        experiment_args (dict): hyper-parameters
        args (dict): system_args + experiment_args
    """
    system_args = {
        # expeirment info
        'project'       : 'NAW-SV',
        'name'          : '2. FT_wavlm_vox2',
        'tags'          : ['NAW-SV'],
        'description'   : '',

        # log
        'path_log'      : '/results',
        'wandb_group'   : '',
        'wandb_entity'  : '',
        'wandb_api_key' : '',

        # dataset,
        'path_train'    : '/[your_data_path]/VoxCeleb2/train',
        'path_test'     : '/[your_data_path]/VoxCeleb1',
        'path_trials'   : '/[your_data_path]/VoxCeleb1/trials',
        'path_musan'    : '/[your_data_path]/musan',
        'path_rir'      : '/[your_data_path]/RIRS_NOISES/simulated_rirs',

        # others
        'num_workers'   : 4,
        'usable_gpu'    : None,
    }

    experiment_args = {
        # huggingface model
        'huggingface_url'           : 'microsoft/wavlm-base-plus',
        
        # experiment
        'epoch'                     : 27,
        'unfreeze_ssl_epoch'        : 20,
        'large_margin_epoch'        : 25,
        'batch_size'                : 24,
        'rand_seed'                 : 1,
        
        # model
        'ssl_hidden_layer_size'     : 768,
        'ssl_hidden_layer_num'      : 12,
        'adapter_hidden_size'       : 64,
        'embed_size'                : 192,
        'masking'                   : True,
        'mask_time_prob'            : 0.05,
        'mask_time_length'          : 10,
        'mask_feature_prob'         : 0.0,
        'mask_feature_length'       : 0,
        'weighted_sum'              : True,
        
        # criterion
        'aam_m'                     : 0.2,
        'aam_s'                     : 30,
        'topk_panelty'              : (5, 0.1),

        # data processing
        'crop_size'                 : 16000 * 3,
        'num_seg'                   : 5,
        'seg_size'                  : 16000 * 3,
        'DA_p'                      : 0.6,
    
        # learning rate
        'lr'                        : 5e-5,
        'lr_min'                    : 5e-7,
		'weight_decay'              : 0,
        'T_mult'                    : 1,
    }

    args = {}
    for k, v in itertools.chain(system_args.items(), experiment_args.items()):
        args[k] = v
    args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))
    args['path_log'] = os.path.join(args['path_log'], args['project'], args['name'])

    return args, system_args, experiment_args