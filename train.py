'''
usage: 
CUDA_VISIBLE_DEVIECS=? python entrance.py --devices 1 --accelerator 'gpu'

... python entrace.py --devices 4 --accelerator gpu --strategy ddp

--fast_dev_run

'''
import os 
import torch 
import argparse
import os.path as osp 
import pytorch_lightning as pl 
from datetime import timedelta
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from data_utils import MultiSpeakerDataset, MultiSpeakerCollate
from model.main import TacotronTrain

def main(args: argparse.Namespace) -> None: 

    cfg = OmegaConf.load(args.cfg)

    dataset = MultiSpeakerDataset(cfg)
    n_total_data = len(dataset)

    trainset, valset = torch.utils.data.random_split(
        dataset, [n_total_data - 20, 20], 
        torch.Generator().manual_seed(114514)       # don't use hash(.) as a seed, because python hash are salted
    )

    tacotron = TacotronTrain(cfg)
    
    os.makedirs(osp.join(cfg.logdir, cfg.runname), exist_ok=True)
    ckpt_manager = ModelCheckpoint(
        save_top_k=5, 
        monitor = 'step', 
        mode = 'max', 
        train_time_interval=timedelta(hours=2)
    )
    trainer = pl.Trainer.from_argparse_args(
        args, 
        callbacks=[ckpt_manager],
        default_root_dir=osp.join(cfg.logdir, cfg.runname),
        gradient_clip_val = cfg.grad_clip_thresh, 
    )

    
    trainer.fit(
        tacotron, 
        ckpt_path = args.restore_ckpt, 
        train_dataloaders = DataLoader(
            trainset, batch_size=cfg.batch_size, collate_fn=MultiSpeakerCollate(), 
            num_workers=cfg.num_workers
        ),
        val_dataloaders = DataLoader(
            valset, batch_size=4, collate_fn=MultiSpeakerCollate()
        )
    )


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('--restore_ckpt', help='resume training from ckpt')
    pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    torch.backends.cudnn.enabled = args.enable_miopen
    
    main(args)

