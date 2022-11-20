import numpy as np
from scipy.io.wavfile import read
import torch
from torch import Tensor, LongTensor
from typing import List, Union, Tuple


def get_mask_from_lengths(lengths):
    '''
    lengths: (bcsz, ) -> (bcsz, max(lengths))
    masked values are `False`
    '''
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)

    # (Tmax, ) < (bcsz, 1) -> (bcsz, Tmax)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


# def to_gpu(x):
#     return x
#     x = x.contiguous()

#     if torch.cuda.is_available():
        # x = x.cuda(non_blocking=True)
    # return torch.autograd.Variable(x)



class TensorPad(): 
    def __init__(self, pad_with=0, return_length: bool = False, length_idx: int = 0) -> None: 
        self.pad_width = pad_with
        self.return_length = return_length
        self.length_idx = length_idx

    def __call__(self, xs: List[Tensor],) -> Tensor: 
        return self.pad_tensors(xs)

    def pad_tensors(self, xs: List[Tensor]) -> Union[Tuple[Tensor, LongTensor], Tensor]: 
        bcsz = len(xs)
        lengths = LongTensor([x.size(self.length_idx) for x in xs])

        _example = xs[0]
        padded_buffer = _example.data.new_ones((bcsz, ) + self.max_size_by_dim(xs)) * self.pad_width
        
        for batch_idx in range(bcsz): 
            piece = xs[batch_idx]
            piece_index = self.get_piece_index(batch_idx, piece.shape)
            padded_buffer[piece_index] = piece
        
        if self.return_length: 
            return padded_buffer, lengths
        else: 
            return padded_buffer

    # for multi-dimension padding in dataloader
    @staticmethod
    def max_size_by_dim(batch_col : List[Tensor]) -> Tuple[int, ...] : 
        '''
        return the maximum length along all dims in a list of tensors
        (have to be matching n_dims or assertion error)
        '''
        def assert_align(all_sizes) : 
            n_dims = Tensor([len(s) for s in all_sizes])
            assert (n_dims - n_dims[0] == 0).all(), f'batch column not aligned : {n_dims}'
            return n_dims[0]
        
        all_sizes  = [b.shape for b in batch_col]
        n_dim      = assert_align(all_sizes)
        return tuple(max(lengths_at_idx) for lengths_at_idx in zip(*all_sizes))

    @staticmethod
    def get_piece_index(batch_idx, piece_shape) : 
        '''
        return a piece's index (range) in the padded buffer (with batch-index as a parameter)
        '''
        return tuple([batch_idx] + [
            slice(piece_len)
            for piece_len in piece_shape
        ])

class TensorStack(): 
    def __init__(self, ) -> None: pass 
    def __call__(self, xs: List[Tensor]) -> Tensor: 
        if xs[0].shape == (1, ): 
            return torch.cat(xs, dim=0)
        else: 
            return torch.stack(xs, dim=0)

