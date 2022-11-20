import os.path as osp
import torch 
import torch.utils.data
import model.layers as layers
import model.utils as utils
import librosa
from text import text_to_sequence
from preprocess_ai3 import MetaLine, fields_iter
from torch import Tensor, LongTensor
from typing import List, Dict, Tuple

ItemType = Tuple[LongTensor, Tensor, LongTensor, Tensor, str]
BatchType = Tuple[Tuple[LongTensor, LongTensor], Tuple[Tensor, LongTensor], LongTensor, Tensor, List[str]]

class MultiSpeakerDataset(torch.utils.data.Dataset): 

    def __init__(self, cfg): 
        self.metalines = self.load_meta(cfg.metafile)
        self.spk2id = self.load_spk2id(cfg.spk2id_file)
        self.base_join_path = cfg.base_join_path

        self.stft = layers.TacotronSTFT(
            cfg.filter_length, cfg.hop_length, cfg.win_length,
            cfg.n_mel_channels, cfg.sampling_rate, cfg.mel_fmin, 
            cfg.mel_fmax
        )

    
    def __len__(self, ) -> int: return len(self.metalines)

    def __getitem__(self, index: int) -> ItemType: 
        '''returns (text, mel, spkid, gate, hans)'''
        _, path, spk, hans, phones = fields_iter(self.metalines[index])
        text_seq = torch.LongTensor(text_to_sequence(phones.split()))
        spkid = torch.LongTensor([self.spk2id[spk]])

        wave = self.load_audio(osp.join(self.base_join_path, path))
        mel  = self.get_mel(wave)
        gate = torch.zeros((len(mel), ))
        gate[-1] = 1

        return text_seq, mel, spkid, gate, hans
    
    def load_meta(self, path: str) -> List[MetaLine]: 
        with open(path) as f: 
            return [
                MetaLine(*l.strip().split('|'))
                for l in f
            ]
    
    def load_spk2id(self, path: str) -> Dict[str, int]: 
        with open(path) as f: 
            return {
                spk: int(idx)
                for spk, idx in (l.strip().split('|') for l in f)
            }

    def load_audio(self, path: str) -> Tensor: 
        wave, _ = librosa.load(path, sr=self.stft.sampling_rate, mono=True)
        wave_trimed, _ = librosa.effects.trim(wave)
        wave = torch.from_numpy(wave_trimed.squeeze()) # (T, )
        return wave

    def get_mel(self, wave: Tensor) -> Tensor: 
        mel = self.stft.mel_spectrogram(wave[None])
        return mel.squeeze(0).T   # (T, nmel)


class MultiSpeakerCollate(): 
    def __init__(self, ) -> None: 
        self.maxpad = utils.TensorPad(0, True, 0)
        self.gatepad = utils.TensorPad(1, False, 0)
        self.stack  = utils.TensorStack()
    def __call__(self, batch: List[ItemType]) -> BatchType: 
        seqs, mels, spkids, gates, hans = zip(*batch)

        padded_seqs, seq_lengths = self.maxpad(seqs)
        _, ids_sorted = torch.sort(seq_lengths, descending=True)

        padded_gates = self.gatepad(gates)
        padded_mels, mel_lengths = self.maxpad(mels)
        padded_mels = padded_mels.permute(0, 2, 1)

        return (
            (padded_seqs[ids_sorted].contiguous(), seq_lengths[ids_sorted].contiguous()), 
            (padded_mels[ids_sorted].contiguous(), mel_lengths[ids_sorted].contiguous()),
            self.stack(spkids)[ids_sorted].contiguous(), 
            padded_gates[ids_sorted].contiguous(), 
            [hans[i] for i in ids_sorted]
        )





