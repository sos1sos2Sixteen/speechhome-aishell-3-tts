import model
import yaml
import torch
from argparse import Namespace

with open('cfg.yaml') as f: cfg = Namespace(**yaml.load(f, yaml.Loader))
net = model.Tacotron2(cfg)

# text_inputs, text_lengths, mels, max_len, output_lengths = in

text_inputs = torch.LongTensor([1,2,3,4,5])[None]
text_lengths = torch.LongTensor([text_inputs.size(1)])
mels = torch.rand((1, 80, 34))
max_len = None
output_lengths = torch.LongTensor([34])
spkids = torch.LongTensor([3])

batch = text_inputs, text_lengths, mels, output_lengths, spkids

r = net(*batch)