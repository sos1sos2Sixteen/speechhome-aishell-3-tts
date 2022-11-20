import argparse
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data_utils import MultiSpeakerDataset, MultiSpeakerCollate


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    dataset = MultiSpeakerDataset(cfg)
    n_total_data = len(dataset)

    trainset, valset = torch.utils.data.random_split(
        dataset, [n_total_data - 20, 20], 
        torch.Generator().manual_seed(114514)       # don't use hash(.) as a seed, because python hash are salted
    )

    loader = DataLoader(
        trainset, batch_size=cfg.batch_size, collate_fn=MultiSpeakerCollate(), 
        num_workers=cfg.num_workers
    )

    for batch in tqdm(loader): pass