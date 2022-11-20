import os
import glob
import torch
import random
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader, Subset
from utils.stft import TacotronSTFT
from utils.utils import read_wav_np


def create_dataloader(hp, args, train):
    dataset = MelFromDisk(hp, args, train)

    if train:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=Subset(dataset, range(200)), batch_size=1, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)


class MelFromDisk(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.path = hp.data.train if train else hp.data.validation
        self.wav_list = glob.glob(os.path.join(self.path, '**', '*.wav'), recursive=True)
        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length + 2
        self.mapping = [i for i in range(len(self.wav_list))]
        self.stft = TacotronSTFT(
            filter_length=hp.audio.filter_length,
            hop_length=hp.audio.hop_length,
            win_length=hp.audio.win_length,
            n_mel_channels=hp.audio.n_mel_channels,
            sampling_rate=hp.audio.sampling_rate,
            mel_fmin=hp.audio.mel_fmin,
            mel_fmax=hp.audio.mel_fmax
        )

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        if self.train:
            idx1 = idx
            idx2 = self.mapping[idx1]
            return self.my_getitem(idx1), self.my_getitem(idx2)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        wavpath = self.wav_list[idx]
        # sr, audio = read_wav_np(wavpath)
        audio, sr = librosa.load(wavpath, sr=self.hp.audio.sampling_rate)
        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio).unsqueeze(0)
        mel = self.stft.mel_spectrogram(audio)
        audio, mel = audio, mel.squeeze(0)

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hp.audio.hop_length
            audio = audio[:, audio_start:audio_start+self.hp.audio.segment_length]

        audio = audio + (1/32768) * torch.randn_like(audio)
        return mel, audio
