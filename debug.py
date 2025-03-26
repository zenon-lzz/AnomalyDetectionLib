"""
=================================================
@Author: Zenon
@Date: 2025-03-26
@Description：Debugger File
==================================================
"""

import os
import time

import numpy as np
import torch.cuda

from tsadlib import ConfigType
from tsadlib import EarlyStopping
from tsadlib import TimesNet
from tsadlib import data_provider
from tsadlib.configs.constants import PROJECT_ROOT

if __name__ == '__main__':

    # Set up device for computation (CUDA GPU, Apple M1/M2 GPU, or CPU)
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'use device: {device}')
    device = torch.device(device)

    # Define paths for dataset and model checkpoints
    # DATASET_ROOT = 'E:\\liuzhenzhou\\datasets'
    DATASET_ROOT = '/Users/liuzhenzhou/Documents/backup/datasets/anomaly_detection/npy'
    # DATASET_ROOT = '/home/lzz/Desktop/datasets'
    DATASET_TYPE = 'MSL'  # Mars Science Laboratory dataset
    CHECKPOINTS = os.path.join(PROJECT_ROOT, 'checkpoints')

    # Configure TimesNet hyperparameters and training settings
    args = ConfigType(**{
        'model': 'TimesNet',
        'root_path': os.path.join(DATASET_ROOT, DATASET_TYPE),
        'dataset': DATASET_TYPE,
        'window_length': 100,  # Length of input sequence
        'batch_size': 128,  # Number of samples per batch
        'num_workers': 0,  # Number of data loading workers
        'top_k': 3,  # Top k time-frequency combinations
        'dimension_model': 8,  # Dimension of model
        'dimension_fcl': 16,  # Dimension of feed-forward network
        'num_kernels': 6,  # Number of inception kernels
        'encoder_layers': 1,  # Number of encoder layers
        'input_channels': 55,  # Input dimension
        'output_channels': 55,  # Output dimension
        'dropout': 0.1,  # Dropout rate
        'embedding_type': 'timeF',  # Time feature embedding type
        'freq': 'h',  # Frequency of the time series (hourly)
        'anomaly_ratio': 1,  # Ratio of anomaly samples
        'train_epochs': 1,  # Number of training epochs
        'learning_rate': 0.0001  # Learning rate for optimization
    })

    # Load training and testing data
    train_data, train_loader = data_provider(args, flag='train')
    test_data, test_loader = data_provider(args, flag='test')

    # Initialize model and training components
    model = TimesNet(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience)
    train_steps = len(train_loader)
    time_now = time.time()

    # Training loop
    for epoch in range(1):
        model.train()
        train_loss = []
        iter_count = 0
        epoch_time = time.time()

        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        # Early stopping check
        if early_stopping(np.mean(train_loss), model, os.path.join(PROJECT_ROOT, 'checkpoints'), 'MSL'):
            print("Early stopping triggered")
            break
