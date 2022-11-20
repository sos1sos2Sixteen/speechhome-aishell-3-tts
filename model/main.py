import torch 
import pytorch_lightning as pl 
import plot
import numpy as np 
from torch import Tensor
from model.model import Tacotron2
from model.layers import TacotronSTFT
from model.loss_function import Tacotron2Loss
from data_utils import BatchType
from typing import Tuple, List


class TacotronTrain(pl.LightningModule): 

    def __init__(self, cfg): 
        super().__init__()

        self.cfg = cfg
        self.tacotron = Tacotron2(cfg)
        self.lossfunc = Tacotron2Loss()

        self.stft = TacotronSTFT(
            cfg.filter_length, cfg.hop_length, cfg.win_length,
            cfg.n_mel_channels, cfg.sampling_rate, cfg.mel_fmin, 
            cfg.mel_fmax
        )
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.tacotron.parameters(), 
            lr = self.cfg.learning_rate, 
            weight_decay=self.cfg.weight_decay
        )

        return optim

    def training_step(self, batch: BatchType, _batch_idx: int) -> Tensor:
        (texts, text_lengths), (mels, mel_lenghts), spkids, gates, _ = batch
        # texts: (bcsz, Tt)
        # text_lenghts: (bcsz, )
        # mels: (bcsz, nmel, Tm)
        # mel_lengths: (bcsz, )
        # spkids: (bcsz, )
        # gates: (bcsz, Tm)

        # mel_before, mel_after, gate_outputs, alignments = r
        r: Tuple[Tensor,...] = self.tacotron(texts, text_lengths, mels, mel_lenghts, spkids)

        loss, (loss_mel_before, loss_mel_after, loss_gate) = self.lossfunc(r, (mels, gates))

        self.log('train/loss-total', loss)

        self.log('train/mel-before', loss_mel_before)
        self.log('train/mel-after',  loss_mel_after)
        self.log('train/gate', loss_gate)

        return loss
    
    def validation_step(self, batch: BatchType, batch_idx: int) -> Tensor: 
        with torch.no_grad(): 
            self.tacotron.eval()
            (texts, text_lengths), (mels, mel_lenghts), spkids, gates, _ = batch

            r = self.tacotron(texts, text_lengths, mels, mel_lenghts, spkids)
            loss, _ = self.lossfunc(r, (mels, gates))

            self.log_val_visualizations(batch_idx, batch, r)

            return loss

    def validation_epoch_end(self, outputs: List[Tensor]) -> None:
        self.log('val/loss-total', torch.tensor(outputs).mean())

    def log_val_visualizations(self, batch_idx: int, batch: BatchType, r: Tuple[Tensor, ...]) -> None: 
        (texts, text_lengths), (mels, mel_lenghts), spkids, gates, hans = batch
        mel_before, mel_after, gate_outputs, alignments = r
        # mel_before/after: (bcsz, nmel, Tm)
        # gate_outputs: (bcsz, Tm)
        # alignments: (bcsz, Tm, Tt)

        batch_ids = [0]
        for idx in batch_ids: 
            real_mel = mels[idx, :, :mel_lenghts[idx]]          # (nmel, Tm')
            pred_mel = mel_after[idx, :, :mel_lenghts[idx]]     # (nmel, Tm')
            pred_gate = gate_outputs[idx, :mel_lenghts[idx]]    # (Tm')
            pred_align = alignments[idx, :mel_lenghts[idx], :text_lengths[idx]] # (Tm', Tt')

            # (Tw', )
            pred_wave = self.stft.inv_mel_spectrogram(pred_mel[None]).squeeze(0)
            real_wave = self.stft.inv_mel_spectrogram(real_mel[None]).squeeze(0)
            
            self.logger.experiment.add_image(
                f'mel/mel-{batch_idx}-{idx}',
                plot.get_data_from_figure(plot.plot_spectrogram(
                    pred_mel.cpu().numpy(), real_mel.cpu().numpy(),
                    f'val/mel-{batch_idx}-{idx}'
                )), 
                dataformats='HWC', global_step=self.global_step
            )

            self.logger.experiment.add_image(
                f'gate/gate-{batch_idx}-{idx}', 
                plot.get_data_from_figure(plot.plot_gate(
                    pred_gate.cpu().numpy(), 
                    f'gate/gate-{batch_idx}-{idx}', 
                )),
                dataformats='HWC', global_step=self.global_step
            )

            self.logger.experiment.add_image(
                f'align/align-{batch_idx}-{idx}', 
                plot.get_data_from_figure(plot.plot_alignment(
                    np.log(pred_align.cpu().numpy().T), 
                    f'align/align-{batch_idx}-{idx}'
                )), 
                dataformats='HWC', global_step=self.global_step
            )

            self.logger.experiment.add_audio(
                f'gt/audio-{batch_idx}-{idx}', 
                real_wave, sample_rate=self.cfg.sampling_rate,
                global_step=self.global_step
            )

            self.logger.experiment.add_audio(
                f'gen/audio-{batch_idx}-{idx}', 
                pred_wave, sample_rate=self.cfg.sampling_rate,
                global_step=self.global_step
            )
    
    def inference(self, text: List[int], spkid: int, use_gl: bool = False): 
        '''```
        ((Tt, ), (1, )) -> (
            mel_before    : (nmel, Tm), 
            mel_after     : (nmel, Tm), 
            gates         : (Tm)
            alignments    : (Tm, Tt)
            optional[wave]: (Tw, )
        )
        ```'''
        with torch.no_grad(): 
            # (1, Tt)
            texts = torch.LongTensor(text)[None].to(self.device)

            # (1, )
            spkids = torch.LongTensor([spkid]).to(self.device)

            r = self.tacotron.inference(texts, spkids)

            returned = tuple(t.squeeze(0) for t in r)
            if use_gl: 
                pred_wave = self.stft.inv_mel_spectrogram(r[1]).squeeze(0)
                returned = returned + (pred_wave, )

            # mel_outputs, mel_outputs_postnet, gate_outputs, alignments, waves
            return tuple(t.cpu() for t in returned)









    






