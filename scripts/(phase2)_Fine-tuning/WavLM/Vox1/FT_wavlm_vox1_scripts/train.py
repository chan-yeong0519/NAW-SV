from tqdm import tqdm
from itertools import cycle

import torch
import torch.distributed as dist

from egg_exp import evaluation
from egg_exp import util

def train_NE(epoch, framework, optimizer, loader, logger, embed_loss_weight, separate_batch=False, masking=False):
    framework.train()
    
    count = 0
    loss_sum = 0
    kd_loss_sum = 0
    embed_kd_loss_sum = 0
    with tqdm(total=len(loader), ncols=90) as pbar:
        for x_clean, x_noise, x_ref, label in loader:
            # Build a mini-batch
            x_noise = torch.cat((x_clean, x_noise), dim=0)
            x_clean = torch.cat((x_clean, x_ref), dim=0)
            
            # to GPU
            x_clean = x_clean.to(dtype=torch.float32, device=framework.device)
            x_noise = x_noise.to(dtype=torch.float32, device=framework.device)

            # clear grad
            optimizer.zero_grad()
            
            # feed forward
            idx_without_adapter = x_clean.size(0) // 2 if separate_batch else None
            kd_loss, embed_kd_loss = framework(x_noise, x_clean=x_clean, idx_without_adapter=idx_without_adapter, masking=masking)

            loss = kd_loss + embed_loss_weight * embed_kd_loss

            # backpropagation
            loss.backward()
            optimizer.step()
            
            # logging
            if logger is not None:
                count += 1
                loss_sum += loss.item()
                kd_loss_sum += kd_loss.item()
                embed_kd_loss_sum += embed_kd_loss.item()
                if len(loader) * 0.02 <= count:
                    logger.log_metric('Loss', loss_sum / count)
                    logger.log_metric('kd_Loss', kd_loss_sum / count)
                    logger.log_metric('embed_kd_Loss', embed_kd_loss_sum / count)
                    loss_sum = 0
                    kd_loss_sum = 0
                    embed_kd_loss_sum = 0
                    count = 0

                desc = f'NE-train-[{epoch}|(loss): {loss.item():.3f}'
                pbar.set_description(desc)
                pbar.update(1)

    _synchronize()

def train_ft(epoch, framework, optimizer, loader, logger, unfreeze_ssl_epoch):
    framework.train()
    
    count = 0
    loss_sum = 0
    with tqdm(total=len(loader), ncols=90) as pbar:
        for x, label in loader:
            # to GPU
            x = x.to(dtype=torch.float32, device=framework.device)
            label = label.to(dtype=torch.int64, device=framework.device)

            # clear grad
            optimizer.zero_grad()
            
            # feed forward
            unfreeze_ssl = True if epoch >= unfreeze_ssl_epoch else False

            _, loss = framework(x, label=label, unfreeze_ssl=unfreeze_ssl)

            loss = loss

            # backpropagation
            loss.backward()
            optimizer.step()
            
            # logging
            if logger is not None:
                count += 1
                loss_sum += loss.item()
                if len(loader) * 0.02 <= count:
                    logger.log_metric('Loss', loss_sum / count)
                    loss_sum = 0
                    count = 0

                desc = f'train-[{epoch}|(loss): {loss.item():.3f}'
                pbar.set_description(desc)
                pbar.update(1)

    _synchronize()

def val(framework, loader, trials):
    framework.eval()
    
    # enrollment
    embeddings_TTA = evaluation.SV_enrollment(framework, loader, use_TTA=True, run_on_ddp=True)

    # EER
    eer = evaluation.test_SV_EER(trials, multi_embedding=embeddings_TTA)
    
    _synchronize()

    return eer

def eval(framework, loader_full, loader_TTA, trials):
    framework.eval()
    
    # enrollment
    embeddings_full = evaluation.SV_enrollment(framework, loader_full, use_TTA=False, run_on_ddp=True)
    embeddings_TTA = evaluation.SV_enrollment(framework, loader_TTA, use_TTA=True, run_on_ddp=True)

    # EER
    eer = evaluation.test_SV_EER(trials, mono_embedding=embeddings_full, multi_embedding=embeddings_TTA)
    
    _synchronize()

    return eer


def noise_eer(framework, loader, trials):
    embeddings = evaluation.SV_enrollment(framework, loader, use_TTA=True, run_on_ddp=True)
    eer = evaluation.test_SV_EER(trials, multi_embedding=embeddings)

    _synchronize()

    return eer

def _synchronize():
    torch.cuda.empty_cache()
    dist.barrier()

