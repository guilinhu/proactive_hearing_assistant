"""
Torch dataset object for synthetically rendered
spatial data
"""
import random

from typing import Tuple
from pathlib import Path

import torch
import numpy as np
import os, glob

import src.utils as utils
from .noise import WhitePinkBrownAugmentation
import torchaudio
from torchmetrics.functional import signal_noise_ratio as snr
from torch.utils.data._utils.collate import default_collate

MAX_LEN = 50

def save_audio_file_torch(file_path, wavform, sample_rate = 16000, rescale = False):
    if rescale:
        wavform = wavform/torch.max(wavform)*0.9
    torchaudio.save(file_path, wavform, sample_rate)

def perturb_amplitude_db(audio, db_change=10):
    random_db = np.random.uniform(-db_change, db_change)
    amplitude_factor = 10 ** (random_db / 20)
    audio = audio * amplitude_factor
    return audio


def scale_to_tgt_pwr(audio: np.ndarray, timestamp, tgt_pwr_dB: float, EPS=1e-9):
    segments = []
    for start_time, end_time in timestamp:
        start_time = max(0, start_time)
        end_time = min(audio.size(-1), end_time)
        
        segment = audio[..., start_time:end_time]
        segments.append(segment)
    
    # Concatenate segments
    concatenated = torch.cat(segments, dim=-1)

    avg_pwr = torch.mean(concatenated**2)
    avg_pwr_dB = 10 * torch.log10(avg_pwr + EPS)
    scale = 10 ** ((tgt_pwr_dB - avg_pwr_dB) / 20)

    audio_scaled = scale * audio
    concatenated_scaled=scale*concatenated

    scaled_pwr_dB = 10 * torch.log10(torch.mean(concatenated_scaled**2) + EPS)


    assert torch.abs(tgt_pwr_dB - scaled_pwr_dB) < 0.1

    return audio_scaled


def scale_utterance(audio, timestamp, rng, db_change=7):
    for start, end in timestamp:
        if rng.uniform(0, 1) < 0.3:
            random_db=rng.uniform(-db_change, db_change)
            amplitude_factor = 10 ** (random_db / 20)
            audio[..., start:end] *= amplitude_factor
        
    return audio
    

def get_snr(target, mixture, EPS=1e-9):
    """
    Computes the average SNR across all channels
    """
    return snr(mixture, target).mean()


def scale_noise_to_snr(target_speech: torch.Tensor, noise: torch.Tensor, target_snr: float):
    """
    Rescales a BINAURAL noise signal to achieve an average SNR (across both channels) equal to target snr.
    Let k be the noise scaling factor
    SNR_tgt = (SNR_left_scaled + SNR_right_scaled) / 2 = 0.5 * (10 log(S_L^T S_L/S_N^T S_N) - 20 log(k) + 10 log(S_R^T S_R / N_R^T N_R) - 20 log(k))
            = 0.5 * (SNR_left_unscaled + SNR_right_unscaled - 40 log(k)) = avg_snr_initial - 20 log (k)
    """
    
    current_snr = get_snr(target_speech, noise + target_speech)

    pwr = (current_snr - target_snr) / 20
    k = 10 ** pwr

    return k * noise


def custom_collate_fn(batch):
    """
    batch: List of tuples (inputs_dict, targets_dict).
    inputs_dict: Dictionary of inputs like 'mixture', 'embed', etc.
    targets_dict: Dictionary of targets like 'target', 'masked_target', etc.
    """
    
    # Separate inputs and targets
    inputs = [item[0] for item in batch]  # item[0] contains the 'inputs' dict
    targets = [item[1] for item in batch]  # item[1] contains the 'targets' dict

    # Process inputs - use default_collate for everything except 'self_timestamp'
    collated_inputs = {}
    for key in inputs[0].keys():
        if key == 'self_timestamp':
            # Handle self_timestamp as a list of lists (variable-length)
            collated_inputs[key] = [item[key] for item in inputs]
        else:
            # For fixed-length tensors, stack them using default_collate
            collated_inputs[key] = default_collate([item[key] for item in inputs])

    # Process targets (normal fixed-length tensors)
    collated_targets = default_collate(targets)

    return collated_inputs, collated_targets


class Dataset(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).

    Each scenario is represented by a folder. Multiple datapoints are generated per
    scenario. This can be customized using the points_per_scenario parameter.
    """
    def __init__(self, input_dir, n_mics=1, sr=8000,
                 sig_len = 30, downsample = 1,
                 split = 'val', output_conversation = 0, 
                 batch_size = 8, 
                 clean_embed=False, 
                 noise_dir = None, 
                 random_audio_length=800,
                 required_first_speaker_as_self_speech=True,
                 spk_emb_exist=True,
                 amplitude_aug_range=0,
                 noise_amplitude_aug_range=7,
                 utter_db_aug=7,
                 input_mean="L",
                 min_snr=-10,
                 max_snr=10,
                 original_val=False,
                 apply_timestamp_aug=False,
                 snr_control=True
                 ):
        super().__init__()

        self.dirs = []
        self.spk_emb_exist=spk_emb_exist
        for _dir in input_dir:
            dir_list = sorted(list(Path(_dir).glob('[0-9]*')))
            for dest in dir_list:
                meta_path = os.path.join(dest, 'metadata.json')
                embed_path = os.path.join(dest, 'embed.pt')
                self_speech_path=os.path.join(dest, 'self_speech.wav')
                
                if self.spk_emb_exist and os.path.exists(meta_path) and os.path.exists(embed_path):
                    self.dirs.append(dest)
                elif not self.spk_emb_exist and os.path.exists(meta_path):
                    self.dirs.append(dest)

        self.noise_dirs = []
        if noise_dir is not None:
            for sub_dir in noise_dir:
                noise_audio_list = glob.glob(os.path.join(sub_dir, '*.wav'))
                if not noise_dir:
                    print("no noise file found")
                self.noise_dirs.extend(noise_audio_list)

            
        self.clean_embed = clean_embed
        self.n_mics = n_mics
        self.sig_len = int(sig_len*sr/downsample)
        self.sr = sr
        self.downsample = downsample
        self.scales = [-3, 3]
        self.output_conversation = output_conversation
        self.apply_timestamp_aug = apply_timestamp_aug

        # Data augmentation
        ### calculate the stat
        self.batch_size = batch_size
        self.split = split
        print(self.split, (len(self.dirs)//batch_size)*batch_size)
        
        self.random_audio_length=random_audio_length
        self.required_first_speaker_as_self_speech=required_first_speaker_as_self_speech
        
        self.amplitude_aug_range=amplitude_aug_range
        self.noise_amplitude_aug_range=noise_amplitude_aug_range
        
        self.pwr_thresh = -60
        self.min_snr=min_snr
        self.max_snr=max_snr
        self.utter_db_aug=utter_db_aug
        self.input_mean=input_mean
        self.original_val=original_val
        self.snr_control=snr_control
        

    def __len__(self) -> int:
        return (len(self.dirs)//self.batch_size)*self.batch_size

    
    def noise_sample(self, noise_file_list, audio_length, rng: np.random.RandomState):
        # NOTE: hardcoded. assume noise is 48k and target is 16k
        # noise_audio=utils.read_audio_file_torch(noise_file, 3)
        
        target_sr = 16000
        
        acc_len=0
        concatenated_audio = None
        while acc_len<=audio_length:
            noise_file=rng.choice(noise_file_list)
            info = torchaudio.info(noise_file)
            noise_sr=info.sample_rate

            noise_wav, _ = torchaudio.load(noise_file)
            if noise_wav.shape[0]>1 and self.input_mean=="L":
                noise_wav=noise_wav[0:1, ...]
            elif noise_wav.shape[0]>1 and self.input_mean=="R":
                noise_wav=noise_wav[1:2, ...]
            elif noise_wav.shape[0]>1 and self.input_mean==True:
                noise_wav=torch.mean(noise_wav, dim=0)
                noise_wav=noise_wav.unsqueeze(0)
        
            if noise_sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=target_sr)
                noise_wav = resampler(noise_wav)
            
            if concatenated_audio is None:
                concatenated_audio = noise_wav
            else:
                concatenated_audio = torch.cat((concatenated_audio, noise_wav), dim=1)
                
            acc_len=concatenated_audio.shape[-1]
            
        
        concatenated_audio=concatenated_audio[..., :audio_length]
            
        assert concatenated_audio.shape[1]==audio_length
        
        return concatenated_audio


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixed_data - M x T
            target_voice_data - M x T
            window_idx_one_hot - 1-D
        """
    
        if self.split == 'train':
            seed = idx + np.random.randint(1000000) 
        else:
            seed = idx
        rng = np.random.RandomState(seed)
        
        curr_dir = self.dirs[idx%len(self.dirs)]
        return self.get_mixture_and_gt(curr_dir, rng)
    
    def diffuse_speech_pattern(self, audio: torch.Tensor, timestamps: list, rng: np.random.RandomState, beta=8000):
        zero_segments = np.array([timestamps[0][0]] + [timestamps[i+1][0] - timestamps[i][1] for i in range(len(timestamps) - 1)] + [audio.shape[-1] - timestamps[-1][1]])
        total_zeros = sum(zero_segments)
    
        # Add noise "diffusion"
        noise = rng.normal(loc=0, scale=beta)
        zero_segments = zero_segments + noise
        
        # Ensure all elements are still positive
        zero_segments[zero_segments <= 0] = 1
        
        # Normalize so that sum is 1
        zero_segments = zero_segments / zero_segments.sum()
        zero_segments = zero_segments * total_zeros
        
        # Floor indices so that we don't exceed audio size
        zero_segments = np.floor(zero_segments).astype(np.int32)
    
        assert zero_segments.sum() <= total_zeros
    
        # Fill in time stamps
        new_audio = torch.zeros_like(audio)
        start_index = 0
        for z, (s, e) in zip(zero_segments[:-1], timestamps):
            start_index += z
            new_audio[..., start_index:start_index+(e-s)] = audio[..., s:e]
            start_index += (e - s)
        
        return new_audio


    def process_audio(self, audio, timestamp, rng, utter_db_aug, tgt_pwr_dB):
        if self.apply_timestamp_aug:
            audio = self.diffuse_speech_pattern(audio, timestamp, rng, beta=16000)
        
        if timestamp==[]:
            return audio
        else:
            audio = scale_to_tgt_pwr(audio, timestamp, tgt_pwr_dB)
            audio=scale_utterance(audio, timestamp, rng, utter_db_aug)
            return audio


    def get_mixture_and_gt(self, curr_dir, rng):        
        metadata2 = utils.read_json(os.path.join(curr_dir, 'metadata.json'))
        
            
        # process self speech
        self_speech = utils.read_audio_file_torch(os.path.join(curr_dir, 'self_speech.wav'), 1, self.input_mean)
        self_speech_original=None
        if os.path.exists(os.path.join(curr_dir, 'self_speech_original.wav')):
            self_speech_original=utils.read_audio_file_torch(os.path.join(curr_dir, 'self_speech_original.wav'), 1, self.input_mean)
            
        self_timestamp=metadata2['target_dialogue'][0]['timestamp']
            
        if self_speech_original is not None:
            list_of_self=[self_speech, self_speech_original]
            concat_self_speech=torch.cat(list_of_self, dim=0)
            utterance_adj_concat_self=scale_utterance(concat_self_speech, self_timestamp, rng, self.utter_db_aug)
            self_speech=utterance_adj_concat_self[0:1, ...]
            self_speech_original=utterance_adj_concat_self[1:2, ...]
        else:
            self_speech=scale_utterance(self_speech, self_timestamp, rng, self.utter_db_aug)

        # process interference speech
        if os.path.exists(os.path.join(curr_dir, f'intereference.wav')):
            interfere = utils.read_audio_file_torch(os.path.join(curr_dir, f'intereference.wav'), 1, self.input_mean)
            scale = 0.8
        else:
            interfers = metadata2["interference"]
            interfere = torch.zeros_like(self_speech)
            if os.path.exists(os.path.join(curr_dir, f'intereference0.wav')):
                for i in range(0, len(interfers)):
                    current_inter=utils.read_audio_file_torch(os.path.join(curr_dir, f'intereference{i}.wav'), 1, self.input_mean)
                    inter_timestamp=metadata2['interference'][i]['timestamp']
                    
                    current_inter=scale_utterance(current_inter, inter_timestamp, rng, self.utter_db_aug)
                    interfere += current_inter
            elif os.path.exists(os.path.join(curr_dir, f'interference0.wav')):
                for i in range(0, len(interfers)):
                    current_inter= utils.read_audio_file_torch(os.path.join(curr_dir, f'interference{i}.wav'), 1, self.input_mean)
                    inter_timestamp=metadata2['interference'][i]['timestamp']
                    
                    current_inter=scale_utterance(current_inter, inter_timestamp, rng, self.utter_db_aug)
                    interfere += current_inter
            scale = 1

        # process other speech
        other_speech = torch.zeros_like(self_speech)
        if self.output_conversation:
            diags = metadata2["target_dialogue"]
            for i in range(len(diags) - 1):
                if os.path.exists(os.path.join(curr_dir, f'target_speech{i}.wav')):
                    wav = utils.read_audio_file_torch(os.path.join(curr_dir, f'target_speech{i}.wav'), 1, self.input_mean)
                    other_timestamp=metadata2['target_dialogue'][i+1]['timestamp']
                    wav=scale_utterance(wav, other_timestamp, rng, self.utter_db_aug)
                    other_speech += wav
                    
                elif os.path.exists(os.path.join(curr_dir, f'other_speech{i}.wav')):
                    wav = utils.read_audio_file_torch(os.path.join(curr_dir, f'other_speech{i}.wav'), 1, self.input_mean)
                    other_timestamp=metadata2['target_dialogue'][i+1]['timestamp']
                    wav=scale_utterance(wav, other_timestamp, rng, self.utter_db_aug)
                    other_speech += wav
                else:
                    raise Exception("no audio file to load")
            
        # add noise, e.g. WHAM
        if self.noise_dirs!=[] and random.random() < 0.3:
            audio_length=interfere.shape[1]
            noise=self.noise_sample(self.noise_dirs, audio_length, rng)
            wham_scale = rng.uniform(0, 1)
            interfere += noise*wham_scale
            
            
        if self_speech_original is not None:
            gt = self_speech_original + other_speech
        else:
            gt = self_speech + other_speech
                
        mixture=gt+interfere
        
        if self.snr_control==True:
            tgt_snr = rng.uniform(self.min_snr, self.max_snr)
            noise = scale_noise_to_snr(gt, mixture - gt, tgt_snr)
            
            mixture = noise + gt
        
        noise_augmentor = WhitePinkBrownAugmentation(
            max_white_level=1e-2,    # Adjust as needed
            max_pink_level=5e-2,     # Adjust as needed
            max_brown_level=5e-2     # Adjust as needed
        )
        
        if self.split=="train" and random.random() < 0.3:
            mixture, gt = noise_augmentor(mixture, gt, rng)
        
        
        reverb_path = os.path.join(curr_dir, f'embed.pt')

        if self.spk_emb_exist:
            embed = torch.load(reverb_path, weights_only=False)
            embed = torch.from_numpy(embed)
        else:
            embed=torch.zeros(256)

        self.output_conversation
        
        input_length=self_speech.shape[1]
        
        start_idx=rng.randint(input_length-self.random_audio_length)
        end_idx=start_idx+self.random_audio_length
        
        # ====peak normalization======
        peak = torch.abs(mixture).max()
        if peak > 1:
            mixture /= peak
            gt /= peak
            self_speech /= peak
            
        
        inputs = {
            'mixture': mixture.float(),
            'embed': embed.float(),
            'self_speech': self_speech[0:1, :].float(),
            'start_idx_list': start_idx,
            'end_idx_list': end_idx
        }
        
        targets = {
            'target': gt[0:1, :].float()
        }
        
        return inputs, targets