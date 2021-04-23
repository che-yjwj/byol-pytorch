# pip install transformers pytorch-lightning
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from model import LitModel
from datamodule import LitDataModule


def set_trainer_args(args):
    if args.freeze:
        frozen = '1'
    else:
        frozen = '0'

    wname = f'CelebA_{args.vision_model}_f{frozen}_s{args.img_size}_lr{args.lr}'
    wandb_logger = WandbLogger(project='CLIP', name=wname, offline=False)
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

    return args



def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--data_dir', default='/workspace/data/face/CelebA_HQ_multi_modal/', type=str)
    parser.add_argument('--text_model', default='distilbert-base-multilingual-cased', type=str)
    parser.add_argument('--vision_model', default='ms_vision', type=str)
    parser.add_argument('--freeze', default=False, type=bool)
    parser.add_argument('--img_size', default=112, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--transformer_embed_dim', default=768, type=int)
    parser.add_argument('--max_len', default=32, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)  
    # parser.add_argument('--lr', type=float, default=1e-1)  # Nan
    # parser.add_argument('--lr', type=float, default=1e-5)  # too low
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--milestones', default=[120,150,180], type=int)
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


    # ------------
    # testing
    # ------------
    # result = trainer.test(data)


if __name__ == '__main__':
    main()

