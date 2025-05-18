"""
=================================================
@Author: Zenon
@Date: 2025-03-11
@Description：The entry file for various tasks execution supported by tsadlib.
==================================================
"""
import argparse
import os.path
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from experiments.exp_benchmarks import BenchmarksExperiment
from tsadlib import constants, logger
from tsadlib.configs.type import ConfigType
from tsadlib.utils.files import write_to_csv
from tsadlib.utils.gpu import empty_gpu_cache

if __name__ == '__main__':

    seed = constants.FIX_SEED

    # sets the seed for the python random module
    random.seed(seed)
    # Set PyTorch's CPU random seed
    torch.manual_seed(seed)
    # Set NumPy random seed
    np.random.seed(seed)

    # CUDA
    if torch.cuda.is_available():
        # Set PyTorch's GPU random seed
        torch.cuda.manual_seed(seed)
        # If multiple Gpus are used, set the random number seed for all Gpus
        torch.cuda.manual_seed_all(seed)

    # mps
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Argument parser for time series anomaly detection model configuration."
    )

    # =========================
    # basic config Parameters
    # =========================
    parser.add_argument('--task_name', type=str, required=True,
                        help=f'task name: options: {constants.TASK_OPTIONS}')
    parser.add_argument('--model', type=str, required=True,
                        help="Model name (e.g., 'TimesNet').")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help="Model's execution status: 'train' or 'test'.")

    # =========================
    # dataset Parameters
    # =========================
    parser.add_argument('--dataset', type=str, required=True,
                        help=f'dataset name: options: {constants.DATASET_OPTIONS}.')
    parser.add_argument('--dataset_root_path', type=str, default='./data',
                        help='root path of various dataset.')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='location of model checkpoints.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training/testing.")
    parser.add_argument('--window_size', type=int, default=100,
                        help="Sequence/window length for time series.")

    # =========================
    # Model Architecture Parameters
    # =========================
    parser.add_argument('--d_model', type=int, required=True,
                        help="Model dimension.")
    parser.add_argument('--input_channels', type=int, required=True,
                        help="Input channel dimension.")
    parser.add_argument('--output_channels', type=int, required=True,
                        help="Output channel dimension.")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate for regularization.")
    parser.add_argument('--dimension_fcl', type=int, default=16,
                        help="Feed-forward layer dimension.")
    parser.add_argument('--encoder_layers', type=int, default=1,
                        help="Number of encoder layers.")
    parser.add_argument('--top_k', type=int, default=3,
                        help="Top k time-frequency combinations (TimesNet).")
    parser.add_argument('--num_kernels', type=int, default=6,
                        help="Number of convolutional kernels.")
    parser.add_argument('--n_heads', type=int, default=8,
                        help="The number of heads in Multiple Head Attention.")
    parser.add_argument('--num_memory', type=int, default=10,
                        help="The number of Memory slots.")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help="The latent space deviation hyperparameter.")
    parser.add_argument('--patch_list', type=int, nargs='+', default=[10, 20],
                        help="List of patch sizes for multi-scale processing.")
    parser.add_argument('--kernel_list', type=int, nargs='+', default=[5],
                        help="List of kernel sizes for multi-scale convolution.")
    parser.add_argument('--hyper_parameter_lambda', type=float, default=0.01,
                        help="Hyperparameter lambda for loss term coefficient.")

    # =========================
    # Optimization Parameters
    # =========================
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument('--runs', type=int, default=5,
                        help="Number of training runs.")
    parser.add_argument('--patience', type=int, default=10,
                        help="Patience for early stopping.")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate for optimizer.")
    parser.add_argument('--end_learning_rate', type=float, default=5e-5,
                        help="End learning rate for optimizer.")
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help="Weight decay for optimizer.")
    parser.add_argument('--warmup_epoch', type=int, default=0,
                        help="Number of warmup epochs.")
    parser.add_argument('--anomaly_ratio', type=float, default=1,
                        help="Expected ratio of anomalies in data.")

    # =========================
    # Hardware Parameters
    # =========================
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help="Whether to use GPU for training/testing.")
    parser.add_argument('--gpu_type', type=str, default='cuda',
                        help="Type of GPU to use: 'cuda' or 'mps'.")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU device index to use.")

    # =========================
    # Other Parameters
    # =========================
    parser.add_argument('--use_tensorboard', default=False, action='store_true',
                        help="Whether to use Tensorboard to record metric.")
    parser.add_argument('--use_wandb', default=False, action='store_true',
                        help="Whether to use wandb to record metric.")


    args_dict = vars(parser.parse_args())
    args = ConfigType(**args_dict)

    if args.task_name == 'benchmarks':
        ExperimentClass = BenchmarksExperiment
    else:
        raise ValueError(f'task name \'{args.task_name}\' does not support.')

    metrics = []
    for run in range(args.runs):

        setting = f'{args.task_name}_{args.model}_{args.dataset}_iter{run + 1}'
        exp = ExperimentClass(args)
        if args.mode == 'train':
            logger.info(f'\n>>>>>>>{run + 1:>3}/{args.runs:>3} start training: >>>>>>>>>>>>>>>>>>>>>>>>>>')
            start_time = time.time()
            exp.train(setting)
            logger.info(f'Training costs time: {time.time() - start_time:10.2}s.')
            logger.info(f'\n>>>>>>>{run + 1:>3}/{args.runs:>3} testing: >>>>>>>>>>>>>>>>>>>>>>>>>>')
            start_time = time.time()
            result = exp.test(setting)
            metrics.append(result)
            logger.info(f'Testing costs time: {time.time() - start_time:10.2}s.')
        else:
            logger.info(f'\n>>>>>>>{run + 1:>3}/{args.runs:>3} testing: >>>>>>>>>>>>>>>>>>>>>>>>>>')
            start_time = time.time()
            result = exp.test(setting)
            metrics.append(result)
            logger.info(f'Testing costs time: {time.time() - start_time:10.2}s.')


        empty_gpu_cache()
        exp.finish()


    df = pd.DataFrame(metrics)
    logger.info(
        f'\n----------------------{args.model} Evaluation Results in {args.dataset} Dataset-----------------------')
    logger.success('All running result:\n{:s}', df.to_string())
    logger.success('Average running result:\n{:s}', df.mean().round(4).to_string())
    result_path = os.path.join('results', args.model, f'{args.dataset}.csv')
    write_to_csv(result_path,
                 f'-----------------------{args.model} Evaluation Results in {args.dataset} Dataset at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-----------------------')
    write_to_csv(result_path, df)
    write_to_csv(result_path, f'Average running result:\n{df.mean().round(4).to_string()}')
