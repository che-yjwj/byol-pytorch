# pip install transformers pytorch-lightning
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from datamodule import LitDataModule
from model import LitModel


def set_trainer_args(args):
    if args.use_momentum:
        framework = 'byol'
    else:
        framework = 'simsiam'

    wname = f'{args.dataset}_{framework}_{args.network}_s{args.img_size}_lr{args.lr}_b{args.batch_size}'
    wandb_logger = WandbLogger(project='BYOL', name=wname, offline=False)
    args.logger = wandb_logger    # W&B integration

    args.max_epochs = args.num_epochs

    args.limit_train_batches = 1.0
    args.limit_val_batches = 1.0
    args.limit_test_batches = 1.0

    args.gpus = -1
    args.distributed_backend = 'ddp'
    # args.gpus = 1

    # gradient clip
    args.gradient_clip_val = 1.0

    # mixed precision
    args.precision = 16
    args.amp_level ='O2'

    # logging
    args.show_progress_bar=True
    args.progress_bar_refresh_rate = 100

    args.checkpoint_callback = True
    args.stochastic_weight_avg = True
    args.resume_from_checkpoint = args.resume

    args.accumulate_grad_batches = 1
    args.sync_batchnorm = True
    args.num_sanity_val_steps = 0

    return args


def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--train_data', default='/workspace/data/cifar/cifar100/train', type=str) # unlabeled_data
    parser.add_argument('--val_data', default='/workspace/data/cifar/cifar100/val', type=str) # labeled_data
    parser.add_argument('--network', default='resnet', type=str)
    parser.add_argument('--use_momentum', default=False, type=bool)
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', type=float, default=3e-4)  
    parser.add_argument('--num_epochs', default=400, type=int)
    parser.add_argument('--milestones', default=[250,300,350], type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ------------
    # setup data
    # ------------
    data = LitDataModule(args)


    # ------------
    # model
    # ------------
    model = LitModel(args)


    # ------------
    # arguments for trainer
    # ------------
    args = set_trainer_args(args)


    # ------------
    # trainer
    # ------------
    trainer = Trainer.from_argparse_args(args)


    # ------------
    # training
    # ------------
    trainer.fit(model, data)


if __name__ == '__main__':
    main()

