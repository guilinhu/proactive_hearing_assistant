import torch
import torch.nn as nn
import src.utils as utils
# from src.models.common.film import FiLM


class FilmLayer(nn.Module):
    def __init__(self, D, C, nF, groups = 1):
        super().__init__()
        self.D = D # speaker dim 256
        self.C = C # latent dim 16
        self.nF = nF
        self.weight = nn.Conv1d(self.D, self.C * nF, 1, groups = groups)
        self.bias = nn.Conv1d(self.D, self.C * nF, 1, groups = groups)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        """
        x: (B, D, F, T)
        embedding: (B, D, F)
        """
        B, D, _F, T = x.shape
        
        w = self.weight(embedding).reshape(B, self.C, _F, 1) # (B, C, F, 1)
        b = self.bias(embedding).reshape(B, self.C, _F, 1) # (B, C, F, 1)
        
        return x * w + b
    
    
class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F]
        """
        x = x.permute(0, 2, 3, 1) # [B, T, F, C]
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2) # [B, C, T, F]
        return x
    
    
class Conv_Emb_Generator(nn.Module):
    def __init__(
        self,
        block_model_name,
        block_model_params,
        spk_dim=256,
        n_srcs=1,
        n_fft=128,
        latent_dim=16,
        num_inputs=1,
        n_layers=6,
        use_first_ln=True,
        n_imics=1,
        lstm_fold_chunk=400,
        E=2,
        use_speaker_emb=True,
        one_emb=True,
        local_context_len=-1
        # 6
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.num_inputs = num_inputs
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.latent_dim = latent_dim
        
        self.use_speaker_emb=use_speaker_emb
        self.one_emb=one_emb
        
        attn_approx_qk_dim=E*n_freqs
        
        self.n_fft = n_fft
        
        self.eps=1.0e-5

        t_ksize = 3
        self.t_ksize = t_ksize
        ks, padding = (t_ksize, t_ksize), (0, 1)
        
        self.n_imics=n_imics
        if not use_speaker_emb:
            self.n_imics=self.n_imics+1
        
        module_list = [nn.Conv2d(2*self.n_imics, latent_dim, ks, padding=padding)]
        
        if use_first_ln:
            module_list.append(LayerNormPermuted(latent_dim))
        
        self.conv = nn.Sequential(
            *module_list
        )

        # FiLM layer
        self.embeds = nn.ModuleList([])
        
        self.local_context_len=local_context_len
        
        self.blocks = nn.ModuleList([])
        for _i in range(n_layers-1):
            self.blocks.append(utils.import_attr(block_model_name)(emb_dim=latent_dim, n_freqs=n_freqs, approx_qk_dim=attn_approx_qk_dim, lstm_fold_chunk=lstm_fold_chunk, last=False, local_context_len=local_context_len, **block_model_params))
        self.blocks.append(utils.import_attr(block_model_name)(emb_dim=latent_dim, n_freqs=n_freqs, approx_qk_dim=attn_approx_qk_dim, lstm_fold_chunk=lstm_fold_chunk, local_context_len=local_context_len, last=True, **block_model_params))
                
        if self.use_speaker_emb and not self.one_emb:
            for _i in range(n_layers-1):
                self.embeds.append(FilmLayer(spk_dim, latent_dim, n_freqs, 1))
        elif self.use_speaker_emb and self.one_emb:
            self.embeds.append(FilmLayer(spk_dim, latent_dim, n_freqs, 1))
    
    def init_buffers(self, batch_size, device):
        conv_buf = torch.zeros(batch_size, 2*self.n_imics, self.t_ksize - 1, self.n_freqs,
                device=device)
            
        deconv_buf = torch.zeros(batch_size, self.latent_dim, self.t_ksize - 1, self.n_freqs,
                                 device=device)

        block_buffers = {}
        for i in range(len(self.blocks)):
            block_buffers[f'buf{i}'] = None

        return dict(conv_buf=conv_buf, deconv_buf=deconv_buf,
                    block_bufs=block_buffers)

    def forward(self, current_input: torch.Tensor, embedding: torch.Tensor, input_state, quantized=False) -> torch.Tensor:
        """
        B: batch, M: mic, F: freq bin, C: real/imag, T: time frame
        D: dimension of the embedding vector
        current_input: (B, CM, T, F)
        embedding: (B, D)
        output: (B, S, T, C*F)
        """
        # [B, C, T, F]
        n_batch, _, n_frames, n_freqs = current_input.shape
        batch = current_input

        if input_state is None:
            input_state = self.init_buffers(current_input.shape[0], current_input.device)
    
        conv_buf = input_state['conv_buf']
        gridnet_buf = input_state['block_bufs']
        
        if quantized:
            batch = nn.functional.pad(batch, (0, 0, self.t_ksize - 1, 0))
        else:
            batch = torch.cat((conv_buf, batch), dim=2)
            
        conv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
        batch = self.conv(batch)  # [B, D, T, F]
        
        if self.use_speaker_emb:
            if not self.one_emb:
                assert len(self.blocks)==self.n_layers
                assert len(self.embeds)==self.n_layers-1
                for ii in range(self.n_layers-1):
                    batch = batch.transpose(2, 3)
                    if ii > 0:
                        batch = self.embeds[ii - 1](batch, embedding) 
                    batch = batch.transpose(2, 3)
                    batch, gridnet_buf[f'buf{ii}'] = self.blocks[ii](batch, gridnet_buf[f'buf{ii}'])
                
                batch = batch.transpose(2, 3)
                batch = self.embeds[-1](batch, embedding)
                batch = batch.transpose(2, 3)
                batch, gridnet_buf[f'buf{self.n_layers-1}'] = self.blocks[self.n_layers-1](batch, gridnet_buf[f'buf{self.n_layers-1}'])
            
            else:
                assert len(self.blocks)==self.n_layers
                assert len(self.embeds)==1
                for ii in range(self.n_layers):
                    batch = batch.transpose(2, 3)
                    if ii == 1:
                        batch = self.embeds[ii - 1](batch, embedding) 
                    batch = batch.transpose(2, 3)
                    batch, gridnet_buf[f'buf{ii}'] = self.blocks[ii](batch, gridnet_buf[f'buf{ii}'])
                    
        else:
            assert len(self.blocks)==self.n_layers
            for ii in range(self.n_layers):
                batch, gridnet_buf[f'buf{ii}'] = self.blocks[ii](batch, gridnet_buf[f'buf{ii}'])    
            
        conversation_emb=batch
            
        return conversation_emb, input_state


    def edge_mode(self):
        for i in range(len(self.blocks)):
            self.blocks[i].edge_mode()

if __name__ == "__main__":
    pass