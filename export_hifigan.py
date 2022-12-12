import argparse
import torch
import json
import os
from hifigan.models import Generator
import os.path as osp


parser = argparse()
parser.add_argument('ckpt', help='path to hifigan generator checkpiont file')
parser.add_argument('--dest', help='directory to dump the exported onnx file', default='exported')
args = parser.parse_args()

with open('hifigan/config_v1.json') as f: 
    h = argparse.Namespace(**json.load(f))
    
vocoder = Generator(h)
vocoder.load_state_dict(torch.load(args.ckpt, map_location='cpu')['generator'])
vocoder.remove_weight_norm()

dummy_input = torch.randn((1, 80, 500))

os.makedirs(args.dest, exist_ok=True)

torch.onnx.export(
    vocoder,
    dummy_input,
    osp.join('exported', 'hifigan.onnx'),
    opset_version=10,
    do_constant_folding=True,
    input_names=["mel"],
    output_names=["wave"],
    dynamic_axes={
        "mel": {2: "mel_seq"},
        "wave": {2: "wave_seq"}
    }
)
