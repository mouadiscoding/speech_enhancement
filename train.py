# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.
# Adapted from https://github.com/NVIDIA/CleanUNet/


import torch
import torch.nn as nn
import numpy as np

import os
import time

from stft_loss import MultiResolutionSTFTLoss
from util import find_max_epoch, print_size
from util import LinearWarmupCosineDecay, loss_fn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
import torch.cuda.amp as amp

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from dataset import load_CleanNoisyPairDataset
from network import CleanUNet

def train_cleanunet_with_val(clean_dir, noisy_dir, batch_size, train_split_size, log_dir, exp_path, ckpt_iter_arg, learning_rate, 
          ell_p, ell_p_lambda, n_iters, 
          stft_lambda, iters_per_valid, iters_per_ckpt):    
    # HEADS-UP: removed stft_config from the params

    # setup local experiment path
    print('exp_path:', exp_path)
    
    # Create tensorboard logger.
    log_directory = os.path.join(log_dir, exp_path)
    tb = SummaryWriter(os.path.join(log_directory, 'tensorboard'))

    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if not os.path.isdir(ckpt_directory):
        os.makedirs(ckpt_directory)
        os.chmod(ckpt_directory, 0o775)
    print("ckpt_directory: ", ckpt_directory, flush=True)

    # load training data
    
    train_dataset, _ = load_CleanNoisyPairDataset(clean_dir, noisy_dir, batch_size=batch_size)
    print('Data loaded')

    train_size = int(train_split_size * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # predefine model
    
    net = CleanUNet(channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4, stride=2,
                 tsfm_n_layers=5, 
                 tsfm_n_head=8,
                 tsfm_d_model=512, 
                 tsfm_d_inner=2048).cuda()
    print_size(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    time0 = time.time()
    if ckpt_iter_arg == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    elif ckpt_iter_arg == 'pretrained':
        ckpt_iter = 'pretrained'
    else:
        ckpt_iter = ckpt_iter_arg

    if ckpt_iter == 'pretrained':
        try:
            # load checkpoint file
            model_path = "./exp/pretrained.pkl"
            checkpoint = torch.load(model_path, map_location='cpu')
            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            print('Pretrained model loaded successfully')
            ckpt_iter = 0
        except:
            print('Failed to load pretrained model. Starting from scratch.')
            ckpt_iter = -1
    elif ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully')
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # training
    n_iter = ckpt_iter + 1

    # define learning rate scheduler and stft-loss
    scheduler = LinearWarmupCosineDecay(
                    optimizer,
                    lr_max=learning_rate,
                    n_iter=n_iters,
                    iteration=n_iter,
                    divider=25,
                    warmup_proportion=0.05,
                    phase=('linear', 'cosine'),
                )

    if stft_lambda > 0:
        mrstftloss = MultiResolutionSTFTLoss().cuda()    # HEADS-UP: removed stft_config from the params
    else:
        mrstftloss = None

    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

    best_val_loss = float('inf')

    # Set up for mixed precision training
    scaler = amp.GradScaler()
    accumulation_steps = 4 # Gradient Accumulation steps
    while n_iter < n_iters + 1:
        # for each epoch
        net.train()
        for clean_audio, noisy_audio in train_loader: 
            
            clean_audio = clean_audio.cuda()
            noisy_audio = noisy_audio.cuda()

            # If you have a data augmentation function augment()
            # noise = noisy_audio - clean_audio
            # noise, clean_audio = augment((noise, clean_audio))
            # noisy_audio = noise + clean_audio
            
            # back-propagation
            optimizer.zero_grad()
            X = (clean_audio, noisy_audio)
            loss, loss_dic = loss_fn(net, X, ell_p, ell_p_lambda, stft_lambda, mrstftloss=mrstftloss)
            reduced_loss = loss.item()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            scheduler.step()
            optimizer.step()

            # output to log
            if n_iter % iters_per_valid == 0:
                net.eval()
                val_losses = []

                with torch.no_grad():
                    for val_clean_audio, val_noisy_audio in val_loader:
                        val_clean_audio = val_clean_audio.cuda()
                        val_noisy_audio = val_noisy_audio.cuda()

                        X_val = (val_clean_audio, val_noisy_audio)
                        val_loss, val_loss_dic = loss_fn(net, X_val, ell_p, ell_p_lambda, stft_lambda, mrstftloss=mrstftloss)
                        val_losses.append(val_loss.item())

                        # Clear GPU memory
                        del val_clean_audio, val_noisy_audio
                        torch.cuda.empty_cache()

                avg_val_loss = sum(val_losses) / len(val_losses)

                print(f"iteration: {n_iter} \ttrain loss: {loss.item():.7f} \tval loss: {avg_val_loss:.7f}", flush=True)

                # print("iteration: {} \treduced loss: {:.7f} \tloss: {:.7f}".format(
                #     n_iter, reduced_loss, loss.item()), flush=True)
                
                # Save validation loss to tensorboard
                tb.add_scalar("Validation/Val-Loss", avg_val_loss, n_iter)

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(ckpt_directory, 'best_model.pkl')
                    torch.save({
                        'iter': n_iter,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'training_time_seconds': int(time.time()-time0)
                    }, best_model_path)
                    print(f'Best model saved at iteration {n_iter} with validation loss {best_val_loss:.7f}')
                
                # Log more detailed metrics if available
                if isinstance(val_loss_dic, dict):
                    for key, value in val_loss_dic.items():
                        tb.add_scalar(f"Validation/{key}", value, n_iter)

                # save to tensorboard
                tb.add_scalar("Train/Train-Loss", loss.item(), n_iter)
                tb.add_scalar("Train/Train-Reduced-Loss", reduced_loss, n_iter)
                tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                tb.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter)

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'iter': n_iter,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time()-time0)}, 
                            os.path.join(ckpt_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1

    # After training, close TensorBoard.
    # tb.flush()
    tb.close()

    return 0