import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .model import Tacotron2
from model.gmm_attention import gmm_parameter_from_mlp_v2
from torch import LongTensor, Tensor
from typing import Tuple

class TracedEncoder(torch.nn.Module): 
    def __init__(self, tac: Tacotron2) -> None: 
        super().__init__()
        self.embedding = tac.embedding
        self.speaker_lut = tac.speaker_lut
        self.encoder = tac.encoder

    def forward(self, texts, text_lengths, spkids) -> Tensor: 
        emb_inputs = self.embedding(texts).transpose(1, 2)
        memory = self.encoder(emb_inputs, text_lengths)
        _, Tmax, _ = memory.shape
        gs = self.speaker_lut(spkids).unsqueeze(1).repeat(1, Tmax, 1)
        aug_memory = torch.cat((memory, gs), dim=-1)
        return aug_memory

def lstmcell2lstm_params(lstm_mod, lstmcell_mod):
    '''move parameteres of a `LSTMCell` instance into a `LSTM` instance.'''
    lstm_mod.weight_ih_l0 = torch.nn.Parameter(lstmcell_mod.weight_ih)
    lstm_mod.weight_hh_l0 = torch.nn.Parameter(lstmcell_mod.weight_hh)
    lstm_mod.bias_ih_l0 = torch.nn.Parameter(lstmcell_mod.bias_ih)
    lstm_mod.bias_hh_l0 = torch.nn.Parameter(lstmcell_mod.bias_hh)
    lstm_mod.flatten_parameters()

def traced_prenet_forward(self, x): 
    '''replace the *always-on* dropout layer in prenet with manual random masking'''
    x1 = x[:]
    for linear in self.layers:
        x1 = F.relu(linear(x1))
        mask = torch.le(torch.rand(256, device=x.device).to(x.dtype), 0.5).to(x.dtype)
        mask = mask.expand(x1.size(0), x1.size(1))
        x1 = x1*mask*2.0

    return x1


def stateless_gmm_attn(
    self, attention_hidden_state, memory, mask, last_mu, last_ctx
    ): 
    """
    stateless version of gmm-attention forward
    PARAMS
    ------
    attention_hidden_state: (bcsz, h): attention rnn last output
    memory: (bcsz, Tt, f): encoder outputs
    mask: (bcsz, Tt): binary mask for padded data
    last_mu: (bcsz, K)
    last_ctx: (bcsz, f)
    """
    s_i = attention_hidden_state    # (bcsz, h)

    if self.use_last_context: 
        # (bcsz, h + memory_dim)
        mlp_input = torch.cat((s_i, last_ctx), dim=1)
    else: 
        # (bcsz, h)
        mlp_input = s_i
    
    # each (bcsz, k)
    Z, W, delta, sigma = gmm_parameter_from_mlp_v2(self.weight_MLP(mlp_input))
    new_mu = last_mu + delta

    _, Tt, _ = memory.shape

    # (bcsz, Tt)
    energy = self.energy_from_parameters((Z, W, new_mu, sigma), Tt)
    if mask is not None: 
        energy = torch.masked_fill(energy, mask, self.score_mask_value)
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    energy = F.softmax(energy, dim=1)                                             # normal smax

    # (bcsz, 1, Tt) x (bcsz, Tt, memory_dim) -> (bcsz, 1, memory_dim) -> (bcsz, memdim)
    next_ctx = torch.bmm(
        energy.unsqueeze(1), memory
    ).squeeze(1)

    self.context_vec = next_ctx

    # 5. returns (bcsz, memory_dim), (bcsz, Tt)
    return next_ctx, energy, new_mu

class TracedDecoderStep(torch.nn.Module): 
    def __init__(self, tac: Tacotron2): 
        super().__init__()

        dec = tac.decoder
        self.dec = dec
        self.p_attention_dropout = dec.p_attention_dropout
        self.p_decoder_dropout = dec.p_decoder_dropout

        self.prenet = dec.prenet
        self.attention_layer = dec.attention_layer
        self.linear_projection = dec.linear_projection
        self.gate_layer = dec.gate_layer

        self.attention_rnn = nn.LSTM(dec.prenet_dim + dec.encoder_embedding_dim,
                                     dec.attention_rnn_dim, 1)
        lstmcell2lstm_params(self.attention_rnn, dec.attention_rnn)


        self.decoder_rnn = nn.LSTM(dec.attention_rnn_dim + dec.encoder_embedding_dim,
                                   dec.decoder_rnn_dim, 1)
        lstmcell2lstm_params(self.decoder_rnn, dec.decoder_rnn)


    def decode(self, decoder_input, in_attention_hidden, in_attention_cell,
               in_decoder_hidden, in_decoder_cell, in_attention_context, memory,
               mask, last_mu):

        cell_input = torch.cat((decoder_input, in_attention_context), -1)

        _, (out_attention_hidden, out_attention_cell) = self.attention_rnn(
            cell_input.unsqueeze(0), (in_attention_hidden.unsqueeze(0),
                                      in_attention_cell.unsqueeze(0)))
        out_attention_hidden = out_attention_hidden.squeeze(0)
        out_attention_cell = out_attention_cell.squeeze(0)

        out_attention_hidden = F.dropout(
            out_attention_hidden, self.p_attention_dropout, False)

        out_attention_context, out_attention_weights, new_mu = stateless_gmm_attn(
            self.attention_layer, 
            out_attention_hidden, memory, mask, last_mu, in_attention_context)

        decoder_input_tmp = torch.cat(
            (out_attention_hidden, out_attention_context), -1)

        _, (out_decoder_hidden, out_decoder_cell) = self.decoder_rnn(
            decoder_input_tmp.unsqueeze(0), (in_decoder_hidden.unsqueeze(0),
                                             in_decoder_cell.unsqueeze(0)))
        out_decoder_hidden = out_decoder_hidden.squeeze(0)
        out_decoder_cell = out_decoder_cell.squeeze(0)

        out_decoder_hidden = F.dropout(
            out_decoder_hidden, self.p_decoder_dropout, False)

        decoder_hidden_attention_context = torch.cat(
            (out_decoder_hidden, out_attention_context), 1)

        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (decoder_output, gate_prediction, out_attention_hidden,
                out_attention_cell, out_decoder_hidden, out_decoder_cell,
                out_attention_weights, out_attention_context, new_mu)

    def forward(self, 
        decoder_input, 
        attention_hidden, 
        attention_cell,
        decoder_hidden, 
        decoder_cell, 
        attention_context, 
        memory, 
        mask,
        last_mu
    ): 
        decoder_input_processed = traced_prenet_forward(self.prenet, decoder_input)
        outputs = self.decode(
            decoder_input_processed, 
            attention_hidden, 
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_context,
            memory,
            mask,
            last_mu
        )
        return outputs


class TracedPostnet(torch.nn.Module):
    def __init__(self, tac: Tacotron2):
        super().__init__()
        self.postnet = tac.postnet

    def forward(self, mel_outputs):
        mel_outputs_postnet = self.postnet(mel_outputs)
        return mel_outputs + mel_outputs_postnet