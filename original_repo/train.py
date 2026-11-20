import argparse
import csv
import datetime
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./new_public_density_data',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.eval_freq < 1:
        raise ValueError("--eval_freq must be positive.")
    if args.start_epoch >= args.epochs:
        raise ValueError("--start_epoch must be less than --epochs.")
    for directory in (args.output_dir, args.checkpoints_dir, args.tensorboard_dir):
        os.makedirs(directory, exist_ok=True)
    protected_paths = [
        os.path.join(args.output_dir, "run_log.txt"),
        os.path.join(args.output_dir, "history.csv"),
        os.path.join(args.checkpoints_dir, "latest.pth"),
        os.path.join(args.checkpoints_dir, "best_mae.pth"),
    ]
    for checkpoint_path in protected_paths:
        if os.path.exists(checkpoint_path):
            raise FileExistsError(
                "Refusing to overwrite existing training output: {}. "
                "Choose new --output_dir and --checkpoints_dir paths."
                .format(checkpoint_path)
            )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The original P2PNet training pipeline requires a CUDA-capable PyTorch "
            "environment; CPU training is not supported."
        )
    # create the logging file
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    with open(run_log_name, "x") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))
    history_path = os.path.join(args.output_dir, 'history.csv')
    history_file = open(history_path, "x", newline="")
    history_writer = csv.DictWriter(
        history_file,
        fieldnames=[
            "epoch",
            "training_loss",
            "validation_loss",
            "validation_mae",
            "validation_rmse",
        ],
    )
    history_writer.writeheader()
    history_file.flush()

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the P2PNet model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set = loading_data(args.data_root)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume the weights and training state if exists
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)
    
    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        validation_loss = evaluate_loss(model, criterion, data_loader_val, device)
        validation_mae = ""
        validation_rmse = ""
        if epoch % args.eval_freq == 0:
            result = evaluate_crowd_no_overlap(model, data_loader_val, device)
            validation_mae, validation_rmse = result[0], result[1]
        history_writer.writerow({
            "epoch": epoch,
            "training_loss": stat["loss"],
            "validation_loss": validation_loss,
            "validation_mae": validation_mae,
            "validation_rmse": validation_rmse,
        })
        history_file.flush()

        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
                log_file.write("loss/loss_ce@{}: {}".format(epoch, stat['loss_ce']))
                
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)
            writer.add_scalar('loss/validation', validation_loss, epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        # change lr according to the scheduler
        lr_scheduler.step()
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
        }, checkpoint_latest_path)
        # Select the best checkpoint using held-out count MAE.
        if epoch % args.eval_freq == 0:
            mae.append(result[0])
            mse.append(result[1])
            # print the evaluation results
            print('=======================================test=======================================')
            print("mae:", result[0], "rmse:", result[1], "best mae:", np.min(mae))
            with open(run_log_name, "a") as log_file:
                log_file.write("mae:{}, rmse:{}, best mae:{}".format(
                    result[0], result[1], np.min(mae)
                ))
            print('=======================================test=======================================')
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, result[0]))
                    log_file.write("metric/mse@{}: {}".format(step, result[1]))
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

            # save the best model since begining
            if result[0] == np.min(mae):
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_mae.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_best_path)
    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    history_file.close()
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)