import math
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet2.torch_utils.get_layer_from_string import get_layer
from torch.nn import init
from torch.nn.parameter import Parameter
import src.utils as utils


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types

        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F]
        """
        x = x.permute(0, 2, 3, 1)  # [B, T, F, C]
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, T, F]
        return x


# Use native layernorm implementation
class LayerNormalization4D(nn.Module):
    def __init__(self, C, eps=1e-5, preserve_outdim=False):
        super().__init__()
        self.norm = nn.LayerNorm(C, eps=eps)
        self.preserve_outdim = preserve_outdim

    def forward(self, x: torch.Tensor):
        """
        input: (*, C)
        """
        x = self.norm(x)
        return x


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        assert len(input_dimension) == 2
        Q, C = input_dimension
        super().__init__()
        self.norm = nn.LayerNorm((Q * C), eps=eps)

    def forward(self, x: torch.Tensor):
        """
        input: (B, T, Q * C)
        """
        x = self.norm(x)

        return x


class LayerNormalization4D_old(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


def mod_pad(x, chunk_size, pad):
    # Mod pad the rminput to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod


class Attention_STFT_causal(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        n_freqs,
        approx_qk_dim=512,
        n_head=4,
        activation="prelu",
        eps=1e-5,
        skip_conn=True,
        use_flash_attention=False,
        dim_feedforward=-1,
        local_context_len=-1,
        # 6
    ):
        super().__init__()
        self.position_code = utils.PositionalEncoding(emb_dim * n_freqs, max_len=5000)

        self.skip_conn = skip_conn
        self.n_freqs = n_freqs
        self.E = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # approx_qk_dim is only approximate
        self.n_head = n_head
        self.V_dim = emb_dim // n_head
        self.emb_dim = emb_dim
        assert emb_dim % n_head == 0
        E = self.E

        self.use_flash_attention = use_flash_attention

        self.local_context_len = local_context_len

        self.add_module(
            "attn_conv_Q",
            nn.Sequential(
                nn.Linear(emb_dim, E * n_head),  # [B, T, Q, HE]
                get_layer(activation)(),
                # [B, T, Q, H, E] -> [B, H, T, Q, E] ->  [B * H, T, Q * E]
                Lambda(
                    lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)
                    .permute(0, 3, 1, 2, 4)
                    .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)
                ),  # (BH, T, Q * E)
                LayerNormalization4DCF((n_freqs, E), eps=eps),
            ),
        )
        self.add_module(
            "attn_conv_K",
            nn.Sequential(
                nn.Linear(emb_dim, E * n_head),
                get_layer(activation)(),
                Lambda(
                    lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)
                    .permute(0, 3, 1, 2, 4)
                    .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)
                ),
                LayerNormalization4DCF((n_freqs, E), eps=eps),
            ),
        )
        self.add_module(
            "attn_conv_V",
            nn.Sequential(
                nn.Linear(emb_dim, (emb_dim // n_head) * n_head),
                get_layer(activation)(),
                Lambda(
                    lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, (emb_dim // n_head))
                    .permute(0, 3, 1, 2, 4)
                    .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * (emb_dim // n_head))
                ),
                LayerNormalization4DCF((n_freqs, emb_dim // n_head), eps=eps),
            ),
        )

        self.dim_feedforward = dim_feedforward

        if dim_feedforward == -1:
            self.add_module(
                "attn_concat_proj",
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    get_layer(activation)(),
                    Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])),
                    LayerNormalization4DCF((n_freqs, emb_dim), eps=eps),
                ),
            )
        else:
            self.linear1 = nn.Linear(emb_dim, dim_feedforward)
            self.dropout = nn.Dropout(p=0.1)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(dim_feedforward, emb_dim)
            self.dropout2 = nn.Dropout(p=0.1)
            self.norm = LayerNormalization4DCF((n_freqs, emb_dim), eps=eps)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def get_lookahead_mask(self, seq_len, device):
        
        if self.local_context_len == -1:
            mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)

            return mask.detach().to(device)

        else:
            mask1 = torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1
            mask2 = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=self.local_context_len) == 0
            mask = (mask1 * mask2).transpose(0, 1)

            return mask.detach().to(device)

    def forward(self, batch):
        ### input/output B T F C
        # attention
        inputs = batch
        B0, T0, Q0, C0 = batch.shape

        # positional encoding
        pos_code = self.position_code(batch)  # 1, T, embed_dim
        _, T, QC = pos_code.shape
        pos_code = pos_code.reshape(1, T, Q0, C0)
        batch = batch + pos_code

        Q = self["attn_conv_Q"](batch)  # [B', T, Q * C]
        K = self["attn_conv_K"](batch)  # [B', T, Q * C]
        V = self["attn_conv_V"](batch)  # [B', T, Q * C]

        emb_dim = Q.shape[-1]

        local_mask = self.get_lookahead_mask(batch.shape[1], batch.device)

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat.masked_fill_(local_mask == 0, -float("Inf"))
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]

        V = torch.matmul(attn_mat, V)  # [B', T, Q*C]
        V = V.reshape(-1, T0, V.shape[-1])  # [BH, T, Q * C]
        V = V.transpose(1, 2)  # [B', Q * C, T]

        batch = V.reshape(B0, self.n_head, self.n_freqs, self.V_dim, T0)  # [B, H, Q, C, T]
        batch = batch.transpose(2, 3)  # [B, H, C, Q, T]
        batch = batch.reshape(B0, self.n_head * self.V_dim, self.n_freqs, T0)  # [B, HC, Q, T]
        batch = batch.permute(0, 3, 2, 1)  # [B, T, Q, C]

        if self.dim_feedforward == -1:
            batch = self["attn_concat_proj"](batch)  # [B, T, Q * C]
        else:
            batch = batch + self._ff_block(batch)  # [B, T, Q, C]
            batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[2] * batch.shape[3])
            batch = self.norm(batch)
        batch = batch.reshape(batch.shape[0], batch.shape[1], Q0, C0)  # [B, T, Q, C])

        # Add batch if attention is performed
        if self.skip_conn:
            return batch + inputs
        else:
            return batch


class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        lstm_fold_chunk,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
        pool="mean",
        last=False,
        local_context_len=-1,
        # 6
    ):
        super().__init__()
        bidirectional = True  # bidirectional within the intra frame lstm

        self.global_atten_causal = True

        self.last = last

        self.pool = pool

        self.lstm_fold_chunk = lstm_fold_chunk
        self.E = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # approx_qk_dim is only approximate

        self.V_dim = emb_dim // n_head
        self.H = hidden_channels
        in_channels = emb_dim * emb_ks
        self.in_channels = in_channels
        self.n_freqs = n_freqs

        ## intra RNN can be optimized by conv or linear because the frequence length are not very large
        self.intra_norm = LayerNormalization4D_old(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True)
        self.intra_linear = nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs

        # inter RNN
        self.inter_norm = LayerNormalization4D_old(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=bidirectional)
        self.inter_linear = nn.ConvTranspose1d(hidden_channels * (bidirectional + 1), emb_dim, emb_ks, stride=emb_hs)

        # attention
        self.pool_atten_causal = Attention_STFT_causal(
            emb_dim=emb_dim,
            n_freqs=n_freqs,
            approx_qk_dim=approx_qk_dim,
            n_head=n_head,
            activation=activation,
            eps=eps,
            local_context_len=local_context_len,
        )

    def _unfold_timedomain(self, x):
        BQ, C, T = x.shape
        x = torch.split(x, self.lstm_fold_chunk, dim=-1)  # [Num_chunk, BQ, C, 100]
        x = torch.cat(x, dim=0).reshape(-1, BQ, C, self.lstm_fold_chunk)  # [Num_chunk, BQ, C, 100]
        x = x.permute(1, 0, 3, 2)  # [BQ, Num_chunk, 100, C]
        return x

    def forward(self, x, init_state=None):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # ===========================Intra RNN start================================
        # define intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)  # [BT, C, Q]

        intra_rnn = torch.split(intra_rnn, self.emb_ks, dim=-1)  # [Q/I, BT, C, I]
        intra_rnn = torch.stack(intra_rnn, dim=0)
        intra_rnn = intra_rnn.permute(1, 2, 3, 0).flatten(1, 2)  # [BT, CI, Q/I]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, nC*emb_ks]
        self.intra_rnn.flatten_parameters()

        # apply intra frame LSTM
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]
        intra_rnn = intra_rnn[:, :, :, :old_Q]  # [B, C, T, Q]
        Q = old_Q
        # ===========================Intra RNN end================================


        # ===========================Inter RNN start================================
        # fold the time domain to chunk
        inter_rnn = self.inter_norm(intra_rnn)  # [B, C, T, F]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)  # [BF, C, T]


        inter_rnn = self._unfold_timedomain(inter_rnn)  ### BQ, NUM_CHUNK, CHUNK_SIZE, C

        BQ, NUM_CHUNK, CHUNKSIZE, C = inter_rnn.shape

        inter_rnn = inter_rnn.reshape(BQ * NUM_CHUNK, CHUNKSIZE, C)  ### BQ* NUM_CHUNK, CHUNK_SIZE, C
        inter_rnn = inter_rnn.transpose(2, 1)  # [B, C, T] 
        input_ = inter_rnn

        inter_rnn = torch.split(inter_rnn, self.emb_ks, dim=-1)

        inter_rnn = torch.stack(inter_rnn, dim=0)
        inter_rnn = inter_rnn.permute(1, 2, 3, 0)

        BF, C, EO, _T = inter_rnn.shape
        inter_rnn = inter_rnn.reshape(BF, C * EO, _T)

        inter_rnn = inter_rnn.transpose(1, 2)

        self.inter_rnn.flatten_parameters()
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn + input_  # [BQ* NUM_CHUNK, C, T]

        inter_rnn = inter_rnn.reshape(B, Q, NUM_CHUNK, C, CHUNKSIZE)
        inter_rnn = inter_rnn.permute(0, 1, 2, 4, 3)  # B, Q, NUM_CHUNK, CHUNKSIZE, C

        input_ = inter_rnn  # B, Q, NUM_CHUNK, CHUNKSIZE, C
        if self.pool == "mean":
            inter_rnn = torch.mean(inter_rnn, dim=3)  # B, Q, NUM_CHUNK, C
        elif self.pool == "max":
            inter_rnn, _ = torch.max(inter_rnn, dim=3)  # B, Q, NUM_CHUNK, C
        else:
            raise ValueError("INvalid pool type!")
        # ===========================Inter RNN end================================

        # ===========================attention start================================
        inter_rnn = inter_rnn.transpose(1, 2)  # B, NUM_CHUNK, Q, C
        inter_rnn = self.pool_atten_causal(inter_rnn)  # B T Q C
        inter_rnn = inter_rnn.transpose(1, 2)  # B Q T C

        if self.last == True:
            return inter_rnn, init_state

        else:
            inter_rnn = inter_rnn.unsqueeze(3)
            inter_rnn = input_ + inter_rnn  # B, Q, NUM_CHUNK, CHUNKSIZE, C

            inter_rnn = inter_rnn.reshape(B, Q, T, C)
            inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # B C T Q
            inter_rnn = inter_rnn[..., :old_T, :]
            # ===========================attention end================================

            return inter_rnn, init_state
