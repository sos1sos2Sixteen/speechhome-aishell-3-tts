import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .layers import ConvNorm, LinearNorm

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        # (bcsz, 2, Tt) -> (bcsz, f, Tt)
        processed_attention = self.location_conv(attention_weights_cat)
        # (bcsz, f, Tt) -> (bcsz, Tt, f)
        processed_attention = processed_attention.transpose(1, 2)
        # (bcsz, Tt, f) -> (bcsz, Tt, f')
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (bcsz, h)
        processed_memory: processed encoder outputs (bcsz, Tt, f')
        attention_weights_cat: cumulative and prev. att weights (bcsz, 2, Tt)

        RETURNS
        -------
        alignment (bcsz, Tt)
        """

        # processed_query: (bcsz, 1, f')
        processed_query = self.query_layer(query.unsqueeze(1))

        # processed_attention_weights: (bcsz, Tt, f')
        processed_attention_weights = self.location_layer(attention_weights_cat)

        # energies: (bcsz, Tt, f) -> (bcsz, Tt, 1)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        # energies: (bcsz, Tt)
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: (bcsz, h): attention rnn last output
        memory: (bcsz, Tt, f): encoder outputs
        processed_memory: (bcsz, Tt, f'): processed encoder outputs
        attention_weights_cat: (bcsz, 2, Tt): previous and cummulative attention weights
        mask: (bcsz, Tt): binary mask for padded data
        """

        # alignement: (bcsz, Tt)
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)

        # (bcsz, 1, Tt) x (bcsz, Tt, f) -> (bcsz, 1, f) -> (bcsz, f)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

