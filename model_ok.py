import torch
import random 
import argparse
from omegaconf import OmegaConf
import model.main


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    tacotron = model.main.TacotronTrain(cfg)


    print(f'train step ...')
    bcsz = 4
    Tt = 20
    Tm = 40
    nmel = 80
    text_lengths = torch.sort(torch.randint(1, Tt, (bcsz, )), descending=True).values
    texts = torch.randint(0, 100, (bcsz, text_lengths.max()))
    mel_lengths = torch.randint(1, Tm-1, (bcsz, ))
    mels = torch.randn(bcsz, nmel, mel_lengths.max())
    spkids = torch.randint(0, 10, (bcsz, ))

    r = tacotron.tacotron(texts, text_lengths, mels, mel_lengths, spkids)
    mel_out, mel_out_postnet, gate_out, alignment = r
    assert mel_out.shape == mel_out_postnet.shape == (bcsz, nmel, mel_lengths.max())
    assert gate_out.shape == (bcsz, mel_lengths.max())
    assert alignment.shape == (bcsz, mel_lengths.max(), text_lengths.max())
    print(f'... OK.')

    print(f'inference step ...')
    tacotron.tacotron.decoder.gate_threshold = torch.inf
    tacotron.tacotron.decoder.max_decoder_steps = 30
    tacotron.inference([random.randint(0, 50) for _ in range(30)], 0, False)

    print(f'... OK.')

    # # texts: (bcsz, Tt)
    # # text_lenghts: (bcsz, )
    # # mels: (bcsz, nmel, Tm)
    # # mel_lengths: (bcsz, )
    # # spkids: (bcsz, )
    # # gates: (bcsz, Tm)

    # # mel_before, mel_after, gate_outputs, alignments = r
    # r: Tuple[Tensor,...] = self.tacotron(texts, text_lengths, mels, mel_lenghts, spkids)

    # loss, (loss_mel_before, loss_mel_after, loss_gate) = self.lossfunc(r, (mels, gates))




