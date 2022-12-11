import argparse
import os
import os.path as osp
import torch 
from omegaconf import OmegaConf
from model.main import TacotronTrain
from model.model import Tacotron2
from model.traced import TracedEncoder, TracedDecoderStep, TracedPostnet

def export_encoder(tacotron: Tacotron2, dest: str) -> None: 
    tencoder = TracedEncoder(tacotron)

    texts = torch.randint(low=0, high=100, size=(1, 50), dtype=torch.long)
    text_lengths = torch.LongTensor([texts.size(1)])
    spkids = torch.LongTensor([0])
    dummy_input = (texts, text_lengths, spkids)

    torch.onnx.export(
        tencoder, 
        dummy_input, 
        osp.join(dest, 'encoder.onnx'),
        opset_version=10,
        do_constant_folding=True,
        input_names=["texts", "text_lengths", "spkids"],
        output_names=["memory"],
        dynamic_axes={
            "texts": {1: "text_seq"},
            "memory": {1: "mem_seq"},
        }
    )



def export_decoder(tacotron: Tacotron2, cfg, dest: str) -> None: 
    tdecoder = TracedDecoderStep(tacotron)

    text_lengths = torch.LongTensor([50])
    memory = torch.randn((1, 50, cfg.encoder_embedding_dim + cfg.speaker_lut_dim))
    decoder_input = tacotron.decoder.get_go_frame(memory)
    mask = torch.zeros((1, 50)).bool()
    attn_hidden = torch.zeros((1, cfg.attention_rnn_dim))
    attn_cell   = torch.zeros_like(attn_hidden)
    dec_hidden = torch.zeros((1, cfg.decoder_rnn_dim))
    dec_cell   = torch.zeros_like(dec_hidden)
    last_mu = torch.zeros((1, cfg.gmm_n_components))
    last_ctx = torch.zeros(1, memory.size(2))

    dummy_input = (
        decoder_input, 
        attn_hidden, attn_cell,
        dec_hidden, dec_cell, 
        last_ctx,
        memory, 
        mask, 
        last_mu
    )

    torch.onnx.export(
        tdecoder,
        dummy_input, 
        osp.join(dest, 'dec_step.onnx'),
        opset_version=10,
        do_constant_folding=True,
        input_names=[
            "decoder_input",
            "attention_hidden",
            "attention_cell",
            "decoder_hidden",
            "decoder_cell",
            "last_ctx",
            "memory",
            "mask",
            "last_mu"
        ],
        output_names=[
            "decoder_output",
            "gate_prediction",
            "out_attention_hidden",
            "out_attention_cell",
            "out_decoder_hidden",
            "out_decoder_cell",
            "out_attention_weights",
            "next_ctx",
            "next_mu"
        ],
        dynamic_axes={
            "memory" : {1: "seq_len"},
            "mask" : {1: "seq_len"},
            "out_attention_weights" : {1: "seq_len"},
    })


def export_postnet(tacotron: Tacotron2, cfg, dest: str) -> None: 
    tpost = TracedPostnet(tacotron)

    dummy_input = torch.randn((1,cfg.n_mel_channels,500))
    torch.onnx.export(
        tpost,
        dummy_input,
        osp.join(dest, 'postnet.onnx'),
        opset_version=10,
        do_constant_folding=True,
        input_names=["mel_before"],
        output_names=["mel_after"],
        dynamic_axes={
            "mel_before": {2: "mel_seq"},
            "mel_after": {2: "mel_seq"}
        }
    )


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt', 
        help='path to gmm-attention based tacotron2 checkpoint', 
        default='logs/ai3_gmm/lightning_logs/version_1/checkpoints/epoch=60-step=79944.ckpt'
    )
    parser.add_argument(
        '--cfg',
        help='path to config file',
        default='configs/cfg_ai3.yaml'
    )
    parser.add_argument(
        '--dest', 
        help='output directory',
        default='exported'
    )

    args = parser.parse_args()

    print(f'to load tacotron weights from: {args.ckpt}')

    cfg = OmegaConf.load(args.cfg)
    assert cfg.use_gmm_attention == True, f'only support GMM attention based Tacotron instances'
    tacotron = TacotronTrain.load_from_checkpoint(args.ckpt, map_location='cpu', cfg=cfg).tacotron

    print(f'weight loaded')


    os.makedirs(args.dest, exist_ok=True)

    export_encoder(tacotron, args.dest); print(f'done exporting encoder')
    export_decoder(tacotron, cfg, args.dest); print(f'doen exporting decoder')
    export_postnet(tacotron, cfg, args.dest); print(f'done exporting postnet')


if __name__ == '__main__': 
    main()

