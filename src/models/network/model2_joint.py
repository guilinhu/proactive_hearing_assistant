import torch
import torch.nn as nn
import src.utils as utils
# from src.models.common.film import FiLM


class FilmLayer(nn.Module):
    def __init__(self, D, C, nF, groups = 1):
        super().__init__()
        self.D = D
        self.C = C
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
    
    
class TSH(nn.Module):
    def __init__(
        self,
        block_model_name,
        block_model_params,
        spk_dim=256,
        latent_dim=48,
        n_srcs=1,
        n_fft=128,
        num_inputs=1,
        n_layers=6,
        use_first_ln=True,
        n_imics=1,
        lstm_fold_chunk=400,
        stft_chunk_size=200,
        latent_dim_model1=16,
        use_speaker_emb=True,
        use_self_speech_model2=True
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.num_inputs = num_inputs
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.latent_dim = latent_dim
        self.lstm_fold_chunk=lstm_fold_chunk
        self.stft_chunk_size=stft_chunk_size
        
        self.n_fft = n_fft
        
        self.eps=1.0e-5

        t_ksize = 3
        self.t_ksize = t_ksize
        ks, padding = (t_ksize, t_ksize), (0, 1)
        
        self.n_imics=n_imics
        
        self.use_self_speech_model2=use_self_speech_model2
        
        if not use_speaker_emb and use_self_speech_model2:
            self.n_imics=self.n_imics+1
        
        module_list = [nn.Conv2d(2*self.n_imics, latent_dim, ks, padding=padding)]
        
        if use_first_ln:
            module_list.append(LayerNormPermuted(latent_dim))
        
        self.conv = nn.Sequential(
            *module_list
        )
        

        # FiLM layer
        self.embeds = nn.ModuleList([])

        # Process through a stack of blocks
        self.blocks = nn.ModuleList([])
        for _i in range(n_layers):
            self.blocks.append(utils.import_attr(block_model_name)(emb_dim=latent_dim, n_freqs=n_freqs, **block_model_params))
                               
        # Project back to TF-Domain
        self.deconv = nn.ConvTranspose2d(latent_dim, n_srcs * 2, ks, padding=( self.t_ksize - 1, 1))
        
        self.latent_dim_model1=latent_dim_model1
        
        if latent_dim_model1!=latent_dim:
            self.projection_layer = nn.Conv2d(latent_dim_model1, latent_dim, kernel_size=1)
    
    def init_buffers(self, batch_size, device):
        conv_buf = torch.zeros(batch_size, 2*self.n_imics, self.t_ksize - 1, self.n_freqs,
                device=device)
            
        deconv_buf = torch.zeros(batch_size, self.latent_dim, self.t_ksize - 1, self.n_freqs,
                                 device=device)

        block_buffers = {}
        for i in range(len(self.blocks)):
            block_buffers[f'buf{i}'] = self.blocks[i].init_buffers(batch_size, device)

        return dict(conv_buf=conv_buf, deconv_buf=deconv_buf,
                    block_bufs=block_buffers)

    def forward(self, current_input: torch.Tensor, embedding: torch.Tensor, input_state, quantized=False) -> torch.Tensor:
        """
        B: batch, M: mic, F: freq bin, C: real/imag, T: time frame
        D: dimension of the embedding vector
        current_input: (B, CM, T, F)
        embedding: (B, D, F)
        output: (B, S, T, C*F)
        """
        
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
        
        embedding=embedding.transpose(1, 3)
            
        for ii in range(self.n_layers):
            if ii==1:
                batch=batch*embedding
            batch, gridnet_buf[f'buf{ii}'] = self.blocks[ii](batch, gridnet_buf[f'buf{ii}'])
        
        deconv_buf = torch.zeros(n_batch, self.latent_dim, self.t_ksize - 1, self.n_freqs,
                                 device=current_input.device)
        if quantized:
            batch = nn.functional.pad(batch, (0, 0, self.t_ksize - 1, 0))
        else:
            batch = torch.cat(( deconv_buf, batch), dim=2)
        
        batch = self.deconv(batch)  # [B, n_srcs*C, T, F]
        
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs]) # [B, n_srcs, 2, n_frames, n_freqs]
        batch = batch.transpose(2, 3).reshape(n_batch, self.n_srcs, n_frames, 2 * n_freqs) # [B, S, T, F]


        input_state['conv_buf'] = conv_buf
        input_state['block_bufs'] = gridnet_buf

        return batch, input_state


    def edge_mode(self):
        for i in range(len(self.blocks)):
            self.blocks[i].edge_mode()

if __name__ == "__main__":
    pass