# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.
# Adapted from https://github.com/NVIDIA/CleanUNet under the MIT License.

from copy import deepcopy
from tqdm import tqdm
from util import rescale, find_max_epoch, print_size, sampling
import os

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import random

from network import CleanUNet
from dataset import load_CleanNoisyPairDataset

from scipy.io.wavfile import write as wavwrite

import numpy as np
from pesq import pesq
from pystoi import stoi
import torch
import pysepm_evo
from pysepm_evo import srmr

def evaluate_model(clean_audio, enhanced_audio, sample_rate=16000):
    """
    Evaluate the speech enhancement model using PESQ, STOI, and SNR metrics.
    
    Args:
    clean_audio (list): List of clean audio signals (numpy arrays)
    enhanced_audio (list): List of enhanced audio signals (numpy arrays)
    sample_rate (int): Sampling rate of the audio signals (default: 16000)
    
    Returns:
    dict: A dictionary containing the average PESQ, STOI, and SNR scores
    """

    pesq_scores = []
    stoi_scores = []
    snr_scores = []
    csig_scores = []
    cbak_scores = []
    covrl_scores = []
    srmr_scores = []

    for clean, enhanced in zip(clean_audio, enhanced_audio):
        
        # Make sure the signals have the same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        # PESQ
        pesq_score = pesq(sample_rate, clean, enhanced, 'wb')  # 'wb' for wideband
        pesq_scores.append(pesq_score)
        
        # STOI
        stoi_score = stoi(clean, enhanced, sample_rate, extended=False)
        stoi_scores.append(stoi_score)
        
        # SNR
        noise = clean - enhanced
        snr_score = 10 * np.log10(np.sum(clean**2) / np.sum(noise**2))
        snr_scores.append(snr_score)

        # Calculate CSIG, CBAK, COVRL
        composite_measures = pysepm_evo.composite(clean, enhanced, sample_rate)
        csig_scores.append(composite_measures[0])
        cbak_scores.append(composite_measures[1])
        covrl_scores.append(composite_measures[2])

        # SRMR score
        srmr_score = srmr(clean, sample_rate)
        srmr_scores.append(srmr_score)

    # Calculate average scores
    avg_pesq = np.mean(pesq_scores)
    avg_stoi = np.mean(stoi_scores)
    avg_snr = np.mean(snr_scores)
    avg_csig = np.mean(csig_scores)
    avg_cbak = np.mean(cbak_scores)
    avg_covrl = np.mean(covrl_scores)
    avg_srmr = np.mean(srmr_scores)

    return {
        'PESQ': avg_pesq,
        'STOI': avg_stoi,
        'SNR': avg_snr,
        'CSIG': avg_csig,
        'CBAK': avg_cbak,
        'COVRL': avg_covrl,
        "SRMR": avg_srmr
    }

def cleanunet_denoise_single_sample(noisy_audio, model_path, sample_rate, output_directory=None, dump=False):
    """
    Denoise a single audio sample.

    Parameters:
    noisy_audio (torch.Tensor):     The noisy audio tensor.
    model_path (str):               Path to the pretrained model checkpoint.
    sample_rate (int):              The sample rate of the audio.
    output_directory (str, optional): Directory to save the denoised audio.
    dump (bool):                    Whether to save the enhanced (denoised) audio.
    """

    # Predefine model
    # net = CleanUNet(channels_input=1, channels_output=1,
    #              channels_H=64, max_H=768,
    #              encoder_n_layers=8, kernel_size=4, stride=2,
    #              tsfm_n_layers=3, 
    #              tsfm_n_head=8,
    #              tsfm_d_model=512, 
    #              tsfm_d_inner=2048).cuda()

    # 5 encoder layers U-Net
    net = CleanUNet(channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4, stride=2,
                 tsfm_n_layers=5, 
                 tsfm_n_head=8,
                 tsfm_d_model=512, 
                 tsfm_d_inner=2048).cuda()
    
    print_size(net)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    noisy_audio = noisy_audio.cuda()

    # Inference
    generated_audio = sampling(net, noisy_audio)

    # Prepare output directory
    if dump and output_directory:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output_directory: ", output_directory, flush=True)

    # Save or return the denoised audio
    if dump and output_directory:
        output_path = os.path.join(output_directory, 'enhanced_audio.wav')
        torchaudio.save(output_path, generated_audio.cpu(), sample_rate)
    else:
        return generated_audio.cpu()
    
def cleanunet_denoise(clean_dir, noisy_dir, batch_size, sample_rate, output_directory, log_dir, exp_path, ckpt_iter, dump=False):
    """
    Denoise audio

    Denoise audio and save the enhanced versions if dump is set to True
    Benchmark the model if dump is set to False

    Parameters:
    clean_dir (str):                directory containing clean audio files (for reference)
    noisy_dir (str):                directory containing noisy audio files to be enhanced
    batch_size (int):               batch size for processing
    sample_rate (int):              sample rate of the audio files
    output_directory (str):         directory to save enhanced audio files
    log_dir (str):                  directory containing model checkpoints
    exp_path (str):                 experiment path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    """

    # setup local experiment path
    print('exp_path:', exp_path)

    _, dataloader = load_CleanNoisyPairDataset(clean_dir, noisy_dir, batch_size=batch_size)
    print('Data loaded')

    # predefine model
    net = CleanUNet(channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4, stride=2,
                 tsfm_n_layers=5, 
                 tsfm_n_head=8,
                 tsfm_d_model=512, 
                 tsfm_d_inner=2048).cuda()
    print_size(net)

    # load checkpoint
    ckpt_directory = os.path.join(log_dir, exp_path, 'checkpoint')
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    
    if ckpt_iter == 'best':
        ckpt_iter = 'best_model'
    elif ckpt_iter != 'pretrained':
        ckpt_iter = int(ckpt_iter)
    
    if ckpt_iter == 'best_model':
        model_path = os.path.join(ckpt_directory, 'best_model.pkl')
        print(f'model path: {model_path}')
    elif ckpt_iter == 'pretrained':
        model_path = "./exp/pretrained.pkl"
    else:
        model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
    
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # get output directory ready
    if ckpt_iter == "pretrained" or "best_model":
        speech_directory = os.path.join(output_directory, exp_path, 'speech', ckpt_iter)
    else:
        speech_directory = os.path.join(output_directory, exp_path, 'speech', '{}k'.format(ckpt_iter//1000))
    if dump and not os.path.isdir(speech_directory):
        os.makedirs(speech_directory)
        os.chmod(speech_directory, 0o775)
    
    if dump:
        print("speech_directory: ", speech_directory, flush=True)

    # inference
    all_generated_audio = []
    all_clean_audio = []
    sortkey = lambda name: '_'.join(name.split('/')[-1].split('_')[1:])
    for idx, (clean_audio, noisy_audio) in enumerate(tqdm(dataloader)):
        noisy_audio = noisy_audio.cuda()
        LENGTH = len(noisy_audio[0].squeeze())
        generated_audio = sampling(net, noisy_audio)
        
        if dump:
            wavwrite(os.path.join(speech_directory, f'enhanced_{idx:04d}.wav'), 
                    sample_rate,
                    generated_audio[0].squeeze().cpu().numpy())
        else:
            all_clean_audio.append(clean_audio[0].squeeze().cpu().numpy())
            all_generated_audio.append(generated_audio[0].squeeze().cpu().numpy())
    
    print(f'Number of generated audio {len(all_generated_audio)* batch_size}')
    if not dump:
        evaluation_results = evaluate_model(all_clean_audio, all_generated_audio, sample_rate)
        print("Evaluation Results:")
        print(f"PESQ: {evaluation_results['PESQ']:.4f}")
        print(f"STOI: {evaluation_results['STOI'] * 100 :.2f}")
        print(f"SNR: {evaluation_results['SNR']:.4f} dB")
        print(f"CSIG: {evaluation_results['CSIG']}")
        print(f"CBAK: {evaluation_results['CBAK']}")
        print(f"COVRL: {evaluation_results['COVRL']}")
        print(f"SRMR: {evaluation_results['SRMR']}")


    return all_clean_audio, all_generated_audio