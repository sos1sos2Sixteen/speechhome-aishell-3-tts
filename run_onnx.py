import argparse
import numpy as np 
import text
import onnxruntime as ort
from tqdm import tqdm

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))
def init_decoder_input(memory): 
    bcsz, Tt, _ = memory.shape

    go_frame = np.zeros((bcsz, 80), dtype=np.float32)
    attn_hidden = np.zeros((bcsz, 1024), dtype=np.float32)
    attn_cell   = np.zeros_like(attn_hidden)
    dec_hidden = np.zeros_like(attn_hidden)
    dec_cell   = np.zeros_like(attn_hidden)
    attn_ctxt    = np.zeros((bcsz, 640), dtype=np.float32)
    mask = np.zeros((bcsz, Tt), dtype=np.bool8)
    last_mu = np.zeros((bcsz, 5), dtype=np.float32)

    return (
        go_frame,
        attn_hidden, 
        attn_cell, 
        dec_hidden,
        dec_cell,
        attn_ctxt,
        mask,
        last_mu
    )

if __name__ == '__main__': 
    print(f'start loading onnx models to runtime')
    enc_session = ort.InferenceSession('exported/encoder.onnx')
    dec_session = ort.InferenceSession('exported/dec_step.onnx')
    pst_session = ort.InferenceSession('exported/postnet.onnx')
    voc_session = ort.InferenceSession('exported/hifigan.onnx')
    print(f'done loading')

    cntxt = '他的著作具有大无畏的尖刻讽刺精神'

    inpt = np.array(text.cn_to_sequence(cntxt))[None, ]
    lengths = np.array([inpt.shape[-1]])
    spkids = np.array([0])


    memory, = enc_session.run(
        None, 
        {
            'texts': inpt, 
            'text_lengths': lengths, 
            'spkids': spkids
        }
    )

    (
        go_frame,
        attn_hidden, 
        attn_cell, 
        dec_hidden,
        dec_cell,
        attn_ctxt,
        mask,
        last_mu
    ) = init_decoder_input(memory)

    mel_history = []
    gate_history = []
    attn_history = []

    dec_input = go_frame
    for idx in tqdm(range(600)): 
        (
            mel_output, 
            gate_output, 
            attn_hidden, 
            attn_cell,
            dec_hidden,
            dec_cell,
            attn_weights,
            attn_ctxt,
            last_mu
        ) = dec_session.run(
            None, {
                "decoder_input": dec_input,
                "attention_hidden": attn_hidden,
                "attention_cell": attn_cell,
                "decoder_hidden": dec_hidden,
                "decoder_cell": dec_cell,
                "last_ctx": attn_ctxt,
                "memory": memory,
                "mask": mask,
                'last_mu': last_mu
            }
        )

        mel_history.append(mel_output)
        gate_history.append(gate_output)
        attn_history.append(attn_weights)

        dec = sigmoid(gate_output) < 0.05
        if dec.sum() == 0: break

        dec_input = mel_output
    
    # (1, nmel, T)
    mel_before = np.concatenate(mel_history, axis=0).T[None]
    mel_after, = pst_session.run(None, input_feed={'mel_before': mel_before})
    wave, = voc_session.run(None, input_feed={'mel': mel_after})
    
