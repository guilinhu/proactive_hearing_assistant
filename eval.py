from src.metrics.metrics import Metrics
import src.utils as utils
import argparse
import os, json, glob
import numpy as np
import torch
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import torch.nn.functional as F
from torchmetrics.functional import signal_noise_ratio as snr


def mod_pad(x, chunk_size, pad):
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod


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


def save_audio_file_torch(file_path, wavform, sample_rate=16000, rescale=False):
    if rescale:
        wavform = wavform / torch.max(wavform) * 0.9
    torchaudio.save(file_path, wavform, sample_rate)


def get_mixture_and_gt(curr_dir, rng, SHIFT_VALUE=0, noise_audio_list=[]):
    metadata2 = utils.read_json(os.path.join(curr_dir, "metadata.json"))
    diags = metadata2["target_dialogue"]

    if os.path.exists(os.path.join(curr_dir, "self_speech.wav")):
        self_speech = utils.read_audio_file_torch(os.path.join(curr_dir, "self_speech.wav"), 1)
    elif os.path.exists(os.path.join(curr_dir, "self_speech_original.wav")):
        self_speech = utils.read_audio_file_torch(os.path.join(curr_dir, "self_speech_original.wav"), 1)

    other_speech = torch.zeros_like(self_speech)

    for i in range(len(diags) - 1):
        wav = utils.read_audio_file_torch(os.path.join(curr_dir, f"target_speech{i}.wav"), 1)
        other_speech += wav

    if os.path.exists(os.path.join(curr_dir, f"intereference.wav")):
        interfere = utils.read_audio_file_torch(os.path.join(curr_dir, f"intereference.wav"), 1)
    else:
        interfere = torch.zeros_like(self_speech)
        interfere += utils.read_audio_file_torch(os.path.join(curr_dir, f"intereference0.wav"), 1)
        interfere += utils.read_audio_file_torch(os.path.join(curr_dir, f"intereference1.wav"), 1)

    gt = self_speech + other_speech
    tgt_snr = rng.uniform(-10, 10)
    interfere = scale_noise_to_snr(gt, interfere, tgt_snr)

    mixture = gt + interfere

    if noise_audio_list != []:
        print("added noise")
        noise_audio = noise_sample(noise_audio_list, mixture.shape[-1], rng)
        wham_scale = rng.uniform(0, 1)
        mixture += noise_audio * wham_scale

    embed_path = os.path.join(curr_dir, "embed.pt")
    if os.path.exists(embed_path):
        embed = torch.load(embed_path, weights_only=False)
        embed = torch.from_numpy(embed)
    else:
        embed = torch.zeros(256)

    L = mixture.shape[-1]

    peak = np.abs(mixture).max()
    if peak > 1:
        mixture /= peak
        self_speech /= peak
        gt /= peak

    inputs = {
        "mixture": mixture.float(),
        "embed": embed.float(),
        "self_speech": self_speech[0:1, :].float(),
    }

    targets = {
        "self": self_speech[0:1, :].numpy(),
        "other": other_speech[0:1, :].numpy(),
        "target": gt[0:1, :].float(),
    }

    return inputs, targets, metadata2


def scale_utterance(audio, timestamp, rng, db_change=7):
    for start, end in timestamp:
        if rng.uniform(0, 1) < 0.3:
            random_db = rng.uniform(-db_change, db_change)
            amplitude_factor = 10 ** (random_db / 20)
            audio[..., start:end] *= amplitude_factor

    return audio


def get_snr(target, mixture, EPS=1e-9):
    """
    Computes the average SNR across all channels
    """
    return snr(mixture, target).mean()


def scale_noise_to_snr(target_speech: torch.Tensor, noise: torch.Tensor, target_snr: float):
    current_snr = get_snr(target_speech, noise + target_speech)

    pwr = (current_snr - target_snr) / 20
    k = 10**pwr

    return k * noise


def run_testcase(model, inputs, device) -> np.ndarray:
    with torch.inference_mode():
        inputs["mixture"] = inputs["mixture"][0:1, ...].unsqueeze(0).to(device)
        inputs["embed"] = inputs["embed"].unsqueeze(0).to(device)
        inputs["self_speech"] = inputs["self_speech"][0:1, ...].unsqueeze(0).to(device)

        inputs["start_idx"] = 0
        inputs["end_idx"] = inputs["mixture"].shape[-1]
        outputs = model(inputs)

        output_target = outputs["output"].squeeze(0)

        final_output = output_target.cpu().numpy()

        return final_output


def get_timestamp_mask(timestamps, mask_shape):
    mask = torch.zeros(mask_shape)
    for s, e in timestamps:
        mask[..., s:e] = 1

    return mask


def noise_sample(noise_file_list, audio_length, rng: np.random.RandomState):
    # NOTE: hardcoded. assume noise is 48k and target is 16k
    target_sr = 16000

    acc_len = 0
    concatenated_audio = None
    while acc_len <= audio_length:
        noise_file = rng.choice(noise_file_list)
        info = torchaudio.info(noise_file)
        noise_sr = info.sample_rate

        noise_wav, _ = torchaudio.load(noise_file)
        noise_wav = noise_wav[0:1, ...]

        if noise_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=target_sr)
            noise_wav = resampler(noise_wav)

        if concatenated_audio is None:
            concatenated_audio = noise_wav
        else:
            concatenated_audio = torch.cat((concatenated_audio, noise_wav), dim=1)

        acc_len = concatenated_audio.shape[-1]

    concatenated_audio = concatenated_audio[..., :audio_length]

    assert concatenated_audio.shape[1] == audio_length

    return concatenated_audio


def main(args: argparse.Namespace):
    device = "cuda" if args.use_cuda else "cpu"

    # Load model
    model = utils.load_torch_pretrained(args.run_dir).model
    model_name = args.run_dir.split("/")[-1]
    model = model.to(device)
    model.eval()

    # Initialize metrics
    snr = Metrics("snr")
    snr_i = Metrics("snr_i")

    si_sdr = Metrics("si_sdr")

    records = []

    noise_audio_list = []
    if args.noise_dir is not None:
        noise_audio_sublist = glob.glob(os.path.join(args.noise_dir, "*.wav"))
        if not noise_audio_sublist:
            print("no noise file found")
        noise_audio_list.extend(noise_audio_sublist)

    for i in range(0, 200):
        rng = np.random.RandomState(i)
        dataset_name = os.path.basename(args.test_dir)
        curr_dir = os.path.join(args.test_dir, "{:05d}".format(i))

        meta_dir = os.path.join(curr_dir, "metadata.json")

        if not os.path.exists(meta_dir):
            continue

        inputs, targets, metadata = get_mixture_and_gt(curr_dir, rng, noise_audio_list=noise_audio_list)

        if inputs is None:
            continue

        self_timestamps = metadata["target_dialogue"][0]["timestamp"]

        target_speech = targets["target"].cpu().numpy()
        row = {"test_case_index": i}
        mixture = inputs["mixture"].cpu().numpy()

        self_speech = inputs["self_speech"].squeeze(0).cpu().numpy()

        inputs["mixture"] = inputs["mixture"][0:1, ...]
        target_speech = target_speech[0:1, ...]

        output_target = run_testcase(model, inputs, device)

        self_timestamps = metadata["target_dialogue"][0]["timestamp"]
        self_mask = get_timestamp_mask(self_timestamps, target_speech.shape)
        self_mask[..., : args.sr] = 0

        if mixture.ndim == 1:
            mixture = mixture[np.newaxis, ...]

        total_input_sisdr = si_sdr(est=mixture[0:1], gt=target_speech, mix=mixture[0:1]).item()
        total_output_sisdr = si_sdr(est=output_target, gt=target_speech, mix=mixture[0:1]).item()

        row[f"sisdr_input_total"] = total_input_sisdr
        row[f"sisdr_output_total"] = total_output_sisdr

        # self

        self_sisdr_mix = si_sdr(
            est=self_mask * mixture[:1], gt=self_mask * target_speech, mix=self_mask * mixture[:1]
        ).item()
        self_sisdr_pred = si_sdr(
            est=self_mask * output_target, gt=self_mask * target_speech, mix=self_mask * mixture[:1]
        ).item()

        row[f"sisdr_mix_self"] = self_sisdr_mix
        row[f"sisdr_pred_self"] = self_sisdr_pred

        # ======other speaker======

        other_timestamps = metadata["target_dialogue"][1]["timestamp"]
        if len(metadata["target_dialogue"]) > 2:
            for j in range(2, len(metadata["target_dialogue"])):
                timestamp = metadata["target_dialogue"][j]["timestamp"]
                other_timestamps = other_timestamps + timestamp

        other_mask = get_timestamp_mask(other_timestamps, target_speech.shape)
        other_mask[..., : args.sr] = 0

        other_sisdr_mix = si_sdr(
            est=other_mask * mixture[:1], gt=other_mask * target_speech, mix=other_mask * mixture[:1]
        ).item()
        other_sisdr_pred = si_sdr(
            est=other_mask * output_target, gt=other_mask * target_speech, mix=other_mask * mixture[:1]
        ).item()

        row[f"sisdr_mix_other"] = other_sisdr_mix
        row[f"sisdr_pred_other"] = other_sisdr_pred

        print(i)
        records.append(row)

        if noise_audio_list != []:
            save_folder = f"./result_{dataset_name}_noise/{model_name}/{i}"
        else:
            save_folder = f"./result_{dataset_name}/{model_name}/{i}"
        os.makedirs(save_folder, exist_ok=True)

        if type(self_speech) == np.ndarray:
            self_speech = torch.from_numpy(self_speech)

        if self_speech.dim() == 1:
            self_speech = self_speech.unsqueeze(0)

        if args.save:
            save_audio_file_torch(
                f"{save_folder}/mix.wav", torch.from_numpy(mixture[0:1]), sample_rate=args.sr, rescale=False
            )
            save_audio_file_torch(f"{save_folder}/self.wav", self_speech, sample_rate=args.sr, rescale=False)
            save_audio_file_torch(
                f"{save_folder}/output_target.wav", torch.from_numpy(output_target), sample_rate=args.sr, rescale=False
            )
            save_audio_file_torch(
                f"{save_folder}/target_speech.wav", torch.from_numpy(target_speech), sample_rate=args.sr, rescale=False
            )

    results_df = pd.DataFrame.from_records(records)

    columns = ["test_case_index"] + [col for col in results_df.columns if col != "test_case_index"]
    results_df = results_df[columns]

    if noise_audio_list != []:
        results_csv_path = f"./result_{dataset_name}_noise/{model_name}_multi.csv"
    else:
        results_csv_path = f"./result_{dataset_name}/{model_name}_multi.csv"
    results_df.to_csv(results_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", type=str, help="Path to test dataset")

    parser.add_argument("run_dir", type=str, help="Path to model run checkpoint")

    parser.add_argument("--sr", type=int, default=16000, help="Project sampling rate")

    parser.add_argument("--noise_dir", type=str, default=None, help="Wham noise directory")

    parser.add_argument("--use_cuda", action="store_true", help="Whether to use cuda")

    parser.add_argument("--save", action="store_true", help="Whether to save output audio")

    main(parser.parse_args())
