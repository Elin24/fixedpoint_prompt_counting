import sys

print(sys.executable)

import os
import time
import argparse
import datetime
import torch
import torch.nn.functional as F
from timm.utils import AverageMeter
import random

from config import get_config
from models import build_model
from datasets import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, plot_curve, set_seed
from einops import rearrange
from torch.cuda.amp import autocast, GradScaler
import random

def get_args_parser():
    parser = argparse.ArgumentParser('Counting Everything training and evaluation script', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main_worker(config):
    opt_level = config.AMP_OPT_LEVEL
    data_loader_train, data_loader_val = build_loader(config.DATA, mode='train'), build_loader(config.DATA, mode='val')

    logger.info(f"Creating model with:{config.MODEL.ENCODER}+{config.MODEL.DECODER}")
    model, criterion = build_model(config.MODEL)
    model.cuda()
    criterion.cuda()

    optimizer = build_optimizer(config, model)
    model_without_ddp = model#.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    max_accuracy = [1e6] * 3

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        mae, mse, loss = validate(config, data_loader_val, model, criterion)
        max_accuracy = (mae, mse, loss)
        logger.info(f"Accuracy of the network on the test images: {mae:.2f} | {mse:.2f}")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    maestack, msestack, lossstack, epostack = [], [], [], []
    scaler = GradScaler()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, scaler)
        if epoch % config.SAVE_FREQ == 0 or epoch > 50:
            mae, mse, loss = validate(config, data_loader_val, model, criterion)
            epostack.append(epoch + 1)
            maestack.append(mae)
            msestack.append(mse)
            lossstack.append(loss)
            plot_curve('mae', epostack, maestack, os.path.join('exp', config.TAG, 'train.log', 'mae_curve.png'))
            plot_curve('mse', epostack, msestack, os.path.join('exp', config.TAG, 'train.log', 'mse_curve.png'))
            plot_curve('loss', epostack, lossstack, os.path.join('exp', config.TAG, 'train.log', 'loss_curve.png'))

            logger.info(f"Average accuracy of the network on the test images: {mae:.2f} | {mse:.2f}")

            # save_checkpoint(config, "last", model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
            if mae * 2 + mse < max_accuracy[0] * 2 + max_accuracy[1]:
                save_checkpoint(config, "best", model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
                max_accuracy = (mae, mse, loss)
            logger.info(f'Min total MAE|MSE|Loss: {max_accuracy[0]:.6f} | {max_accuracy[1]:.2f} | {max_accuracy[2] * 1e5:.2f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    # model.decoder.encoderlayer.tau *= 0.98
    for idx, (samples,targets, boxmaps, potmaps, clip_masks) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        boxmaps = boxmaps.cuda(non_blocking=True)
        potmaps = potmaps.cuda(non_blocking=True)
        clip_masks = clip_masks.cuda(non_blocking=True)


        # resize image to avoid overfitting
        rscale = random.random() * 0.6 + 0.7
        b, _, h, w = samples.shape
        nh, nw = int(h * rscale), int(w * rscale)
        nh, nw = nh + (16 - nh % 16), nw + (16 - nw % 16)
        samples = F.interpolate(samples, (nh, nw), mode='bilinear', align_corners=False)

        boxmaps = F.adaptive_max_pool2d(boxmaps, (nh // 16, nw // 16))
        potmaps = F.adaptive_max_pool2d(potmaps, (nh // 16, nw // 16))
        clip_masks = F.adaptive_max_pool2d(clip_masks, (nh // 16, nw // 16))
        prompts = torch.cat((boxmaps, potmaps, clip_masks), dim=1)

        prompt_select = torch.rand((b, 3, 1, 1)).to(boxmaps)
        prompt_select = prompt_select == prompt_select.max(dim=1, keepdim=True).values
        prompts = (prompts * prompt_select).sum(dim=1, keepdim=True)


        mixflag = True
        with autocast(enabled=mixflag):
            posden, negdens, pdemap, ndemaps = model(samples, targets, prompts)
            posloss = criterion(posden, tarden=pdemap.detach()) + criterion(pdemap, tardot=targets) 
            negloss = criterion(negdens, tarden=ndemaps.detach()) + criterion(ndemaps, tardot=torch.zeros_like(ndemaps))
            loss = posloss + negloss
        
        if mixflag:
            scaler.scale(loss).backward()
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)

                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    grad_norm = None
                    logger.info(f"grad_norm is nan or inf. Ignore this batch.")
                else:
                    scaler.step(optimizer)
                    
                scaler.update()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                grad_norm = None
        else:
            loss.backward()
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    grad_norm = None
                    logger.info(f"grad_norm is nan or inf. Ignore this batch")
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                grad_norm = None

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None: norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr*1e5:.3f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val * 1e3:.3f} ({loss_meter.avg * 1e3:.3f})\t'
                f'grad_norm {norm_meter.val * 1e3:.3f} ({norm_meter.avg * 1e3:.3f})\t'
                f'mem {memory_used:.0f}MB\t'
                )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = [AverageMeter() for _ in range(3)]
    mae_meter = [AverageMeter() for _ in range(3)]
    mse_meter = [AverageMeter() for _ in range(3)]
    titles = ['box', 'point', 'clip']

    end = time.time()
    for idx, (images, target, boxmaps, potmaps, clip_masks) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        boxmaps = boxmaps.cuda(non_blocking=True)
        potmaps = potmaps.cuda(non_blocking=True)
        clip_masks = clip_masks.cuda(non_blocking=True)

        bsize = target.size(0)
        # compute output
        
        with torch.no_grad():
            for i, masks in enumerate([boxmaps, potmaps, clip_masks]):
        
                nh, nw = images.shape[-2:]
                boxmaps = F.adaptive_max_pool2d(masks, (nh // 16, nw // 16))
                denmap = model(images, boxmaps=boxmaps).relu()
                loss = criterion(denmap, tardot=target)
                prednum = (denmap / config.MODEL.FACTOR).sum(dim=(1,2,3))
                tarnum = target.sum(dim=(1,2,3))
                
                diff = torch.abs(prednum - tarnum)
                mae, mse = diff.mean(), (diff ** 2).mean()

                loss_meter[i].update(loss.item(), bsize)
                mae_meter[i].update(mae.item(), bsize)
                mse_meter[i].update(mse.item(), bsize)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            for i, title in enumerate(titles):
                logger.info(
                    f'{title}: [{idx}/{len(data_loader)}]  '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    f'Loss {loss_meter[i].val:.6f} ({loss_meter[i].avg:.6f})  '
                    f'MAE {mae_meter[i].val:.3f} ({mae_meter[i].avg:.3f})  '
                    f'MSE {mse_meter[i].val ** 0.5:.3f} ({mse_meter[i].avg ** 0.5:.3f})  '
                    f'Mem {memory_used:.0f}MB')
    for i, title in enumerate(titles):
        logger.info(f'{title} * MAE {mae_meter[i].avg:.3f} MSE {mse_meter[i].avg ** 0.5:.3f}')
    return sum([mae.avg for mae in mae_meter]) / 3,  \
        sum([mse.avg ** 0.5 for mse in mse_meter]) / 3, \
    sum([loss.avg for loss in loss_meter]) / 3


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

if __name__ == '__main__':
    #torch.cuda.set_per_process_memory_fraction(0.5, 0)
    _, config = get_args_parser()

    
    torch.cuda.set_device('cuda:0')
    set_seed(config.SEED)

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        config.defrost()
        config.TRAIN.BASE_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.WARMUP_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.MIN_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main_worker(config)
