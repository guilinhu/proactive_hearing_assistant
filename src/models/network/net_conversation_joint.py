import torch
import torch.nn as nn

from .model1 import Conv_Emb_Generator
from .model2_joint import TSH
import torch.nn.functional as F
import numpy as np
import copy


def mod_pad(x, chunk_size, pad):
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod


# A TF-domain network guided by an embedding vector
class Net_Conversation(nn.Module):
    def __init__(self, 
                 model1_block_name, 
                 model1_block_params,
                 model2_block_name,
                 model2_block_params,
                 stft_chunk_size=64, 
                 stft_pad_size=32, 
                 stft_back_pad=32,
                 num_input_channels=1, 
                 num_output_channels=1, 
                 num_sources=1,
                 speaker_embed = 256,
                 num_layers_model1=3,
                 num_layers_model2=3,
                 latent_dim_model1=16,
                 latent_dim_model2=32,
                 use_sp_feats=False, 
                 use_first_ln=True,
                 n_imics=1,
                 window="hann",
                 lstm_fold_chunk=400,
                 E=2,
                 use_speaker_emb_model1=True,
                 one_emb_model1=True,
                 use_self_speech_model2=True,
                 local_context_len=-1
                 ):
        super(Net_Conversation, self).__init__()

        assert num_sources == 1

        # num input/output channels        
        self.nI = num_input_channels
        self.nO = num_output_channels

        # num channels to the TF-network
        num_separator_inputs = self.nI * 2 + use_sp_feats * (3 * (self.nI - 1))

        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.stft_back_pad = stft_back_pad
        self.n_srcs = num_sources
        self.use_sp_feats = use_sp_feats
        
        # Input conv to convert input audio to a latent representation        
        self.nfft = stft_back_pad + stft_chunk_size + stft_pad_size
        
        self.nfreqs = self.nfft//2 + 1
        
        self.lstm_fold_chunk=lstm_fold_chunk

        # Construct synthesis/analysis windows (rect)
        if window=="hann":
            window_fn = lambda x: np.hanning(x)
        elif window=="rect":
            window_fn = lambda x: np.ones(x)
        else:
            raise ValueError("Invalid window type!")
        
        if ((stft_pad_size) % stft_chunk_size) == 0:
            print("Using perfect STFT windows")
            self.analysis_window = torch.from_numpy(window_fn(self.nfft)).float()

            # eg. inverse SFTF
            self.synthesis_window = torch.zeros(stft_pad_size + stft_chunk_size).float()
            
            A = self.synthesis_window.shape[0]
            B = self.stft_chunk_size
            N = self.analysis_window.shape[0]
            
            assert (A % B) == 0
            for i in range(A):
                num = self.analysis_window[N - A + i]
            
                denom = 0
                for k in range(A//B):
                    denom += (self.analysis_window[N - A + (i % B) + k * B] ** 2)
                
                self.synthesis_window[i] = num / denom
        else:
            print("Using imperfect STFT windows")
            self.analysis_window = torch.from_numpy( window_fn(self.nfft) ).float()
            self.synthesis_window = torch.from_numpy( window_fn(stft_chunk_size + stft_pad_size) ).float()
        
        self.istft_lookback = 1 + (self.synthesis_window.shape[0] - 1) // self.stft_chunk_size
        
        if local_context_len!=-1:
            local_context_len=local_context_len//stft_chunk_size//lstm_fold_chunk

        self.model1 = Conv_Emb_Generator(
            model1_block_name,
            model1_block_params,
            spk_dim = speaker_embed,
            latent_dim = latent_dim_model1,
            n_srcs = num_output_channels * num_sources,
            n_fft = self.nfft,
            num_inputs = num_separator_inputs,
            n_layers = num_layers_model1,
            use_first_ln=use_first_ln,
            n_imics=n_imics,
            lstm_fold_chunk=lstm_fold_chunk,
            E=E,
            use_speaker_emb=use_speaker_emb_model1,
            one_emb=one_emb_model1,
            local_context_len=local_context_len
        )

        self.quantized = False
        
        self.use_self_speech_model2=use_self_speech_model2
        
        self.model2=TSH(
            model2_block_name,
            model2_block_params,
            spk_dim = speaker_embed,
            latent_dim = latent_dim_model2,
            latent_dim_model1=latent_dim_model1,
            n_srcs = num_output_channels * num_sources,
            n_fft = self.nfft,
            num_inputs = num_separator_inputs,
            n_layers = num_layers_model2,
            use_first_ln=use_first_ln,
            n_imics=n_imics,
            lstm_fold_chunk=lstm_fold_chunk,
            stft_chunk_size=stft_chunk_size,
            use_speaker_emb=use_speaker_emb_model1,
            use_self_speech_model2=use_self_speech_model2
        )
        
        self.use_speaker_emb_model1=use_speaker_emb_model1

    def init_buffers(self, batch_size, device):
        buffers = {}
        
        buffers['model1_bufs'] = self.model1.init_buffers(batch_size, device)
        
        buffers['model2_bufs'] = self.model2.init_buffers(batch_size, device)

        buffers['istft_buf'] = torch.zeros(batch_size * self.n_srcs * self.nO,
                                           self.synthesis_window.shape[0],
                                           self.istft_lookback, device=device)

        return buffers

    # compute STFT
    def extract_features(self, x):
        """
        x: (B, M, T)
        returns: (B, C*M, T, F)
        """
        B, M, T = x.shape

        x = x.reshape(B*M, T)
        x = torch.stft(x, n_fft = self.nfft, hop_length = self.stft_chunk_size,
                          win_length = self.nfft, window=self.analysis_window.to(x.device),
                          center=False, normalized=False, return_complex=True)
        
        x = torch.view_as_real(x) # [B*M, F, T, 2]
        BM, _F, T, C = x.shape

        x = x.reshape(B, M, _F, T, C) # [B, M, F, T, 2]

        x = x.permute(0, 4, 1, 3, 2) # [B, 2, M. T, F]
        
        x = x.reshape(B, C*M, T, _F)

        return x

    def synthesis(self, x, input_state):
        """
        x: (B, S, T, C*F)
        returns: (B, S, t) 
        """
        istft_buf = input_state['istft_buf']
        
        x = x.transpose(2, 3) # [B, S, CF, T]

        B, S, CF, T = x.shape
        X = x.reshape(B*S, CF, T)
        X = X.reshape(B*S, 2, -1, T).permute(0, 2, 3, 1) # [BS, F, T, C]
        X = X[..., 0] + 1j * X[..., 1]

        x = torch.fft.irfft(X, dim=1) # [BS, iW, T]
        x = x[:, -self.synthesis_window.shape[0]:] # [BS, oW, T]

        # Apply synthesis window
        x = x * self.synthesis_window.unsqueeze(0).unsqueeze(-1).to(x.device)

        oW = self.synthesis_window.shape[0]
        
        # Concatenate blocks from previous IFFTs
        x = torch.cat([istft_buf, x], dim=-1)
        istft_buf = x[..., -istft_buf.shape[1]:] # Update buffer

        # Get full signal
        x = F.fold(x, output_size=(self.stft_chunk_size * x.shape[-1] + (oW - self.stft_chunk_size), 1),
                      kernel_size=(oW, 1), stride=(self.stft_chunk_size, 1)) # [BS, 1, t]
        
        x = x[:, :, -T * self.stft_chunk_size - self.stft_pad_size: - self.stft_pad_size] 
        x = x.reshape(B, S, -1) # [B, S, t]

        input_state['istft_buf'] = istft_buf

        return x, input_state


    def predict_model1(self, x, input_state, speaker_embedding, pad=True):
        """
        B: batch
        M: mic
        t: time step (time-domain)
        x: (B, M, t)
        R: real or imaginary
        """

        mod = 0
        if pad:
            pad_size = (self.stft_back_pad, self.stft_pad_size)
            x, mod = mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)
            
        # Time-domain to TF-domain
        x = self.extract_features(x) # [B, RM, T, F]
        
        if speaker_embedding is not None:
            speaker_embedding=speaker_embedding.unsqueeze(2)
        
        conversation_emb, input_state['model1_bufs'] = self.model1(x, speaker_embedding, input_state['model1_bufs'], self.quantized)
        
        return conversation_emb, input_state
    
    def predict_model2(self, x, conversation_emb, input_state, pad=True):
        """
        B: batch
        M: mic
        t: time step (time-domain)
        x: (B, M, t)
        R: real or imaginary
        """
        mod = 0
        if pad:
            pad_size = (self.stft_back_pad, self.stft_pad_size)
            x, mod = mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)
            
        x = self.extract_features(x)
            
        x, input_state['model2_bufs']=self.model2(x, conversation_emb, input_state['model2_bufs'], self.quantized)
        
        # TF-domain to time-domain
        x, next_state = self.synthesis(x, input_state) # [B, S * M, t]
        
        if mod != 0:
            x = x[:, :, :-mod]

        return x, next_state
    

    def forward(self, inputs, input_state = None, pad=True):
        x = inputs['mixture']
        
        start_idx_input=inputs['start_idx']
        end_idx_input=inputs['end_idx']
        
        assert ((end_idx_input - start_idx_input) % self.stft_chunk_size) == 0
        
        # Snap start and end to chunk
        start_idx_input = (start_idx_input // self.stft_chunk_size) * self.stft_chunk_size
        end_idx_input = (end_idx_input // self.stft_chunk_size) * self.stft_chunk_size
        
        B, M, t=x.shape
        
        audio_range=torch.tensor([start_idx_input, end_idx_input]).to(x.device)
        audio_range = audio_range.unsqueeze(0).repeat(B, 1)
        
        spk_embed = inputs['embed']
        self_speech=None
        
        if not self.use_speaker_emb_model1:
            self_speech=inputs['self_speech']

            combined_audio = torch.cat((x, self_speech), dim=1)
            x=combined_audio

        if input_state is None:
            input_state = self.init_buffers(x.shape[0], x.device)
        
        B, M, t = x.shape
        
        # enter slow model
        conversation_emb, input_state = self.predict_model1(x, input_state, spk_embed, pad=pad) # [B, F, T, C]
        
        # slice conv embedding and corresponding audio
        B, _F, T, C = conversation_emb.shape
        conversation_emb = conversation_emb.permute(0, 1, 3, 2) # [B, F, C, T]
        conversation_emb = torch.roll(conversation_emb, 1, dims=-1)
        conversation_emb[..., 0] = 0
        conversation_emb = conversation_emb.flatten(0,3).unsqueeze(1) # [*, 1]
        multiplier = torch.tile(conversation_emb, (1, self.lstm_fold_chunk)) # [*, L]
        multiplier = multiplier.reshape(B, _F, C, T, self.lstm_fold_chunk).flatten(3,4) # [B, F, C, T*L]
        multiplier = multiplier.permute(0, 1, 3, 2) # [B, F, T*L, C]
        
        slicing_length=end_idx_input-start_idx_input+self.stft_back_pad+self.stft_pad_size
        
        padded_start=start_idx_input-self.stft_back_pad
        padded_end=end_idx_input+self.stft_pad_size
        
        pad_left=max(-padded_start, 0)
        pad_right=max(padded_end-t, 0)
        
        actual_start=max(padded_start, 0)
        actual_end=min(padded_end, t)
        
        if self.use_self_speech_model2:
            sliced_x=x[:, :, actual_start:actual_end]
        else:
            x_no_self_speech=inputs["mixture"]
            sliced_x=x_no_self_speech[:, :, actual_start:actual_end]
        
        padding = (pad_left, pad_right, 0, 0, 0, 0)
        
        sliced_x=F.pad(sliced_x, padding, "constant", 0)
        
        converted_start_idx=start_idx_input//self.stft_chunk_size
        converted_end_idx=end_idx_input//self.stft_chunk_size
        
        sliced_emb=multiplier[:, :, converted_start_idx:converted_end_idx, :]
        
        assert sliced_x.shape[2]==slicing_length
        assert sliced_emb.shape[2]==(slicing_length-self.stft_back_pad-self.stft_pad_size)//self.stft_chunk_size
        
        model2_output, input_state = self.predict_model2(sliced_x, sliced_emb, input_state, pad=False)
        model2_output = model2_output.reshape(B, self.n_srcs, self.nO, model2_output.shape[-1])
        
        return {'output': model2_output[:, 0], 'next_state': input_state, 'audio_range': audio_range}
        

if __name__ == "__main__":
    pass