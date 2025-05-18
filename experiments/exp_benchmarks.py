"""
=================================================
@Author: Zenon
@Date: 2025-05-01
@Description：Benchmarks Experiment for Various Models
==================================================
"""
import os.path
import time
import warnings

import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experiments.exp_basic import ExperimentBase
from tsadlib import ConfigType, DatasetSplitEnum, ValidateMetricEnum, EarlyStoppingModeEnum, ThresholdWayEnum
from tsadlib import logger
from tsadlib.data_provider.data_factory import data_provider
from tsadlib.metrics.anomaly_metrics import AnomalyMetrics
from tsadlib.utils.learning_rate_decay import PolynomialDecayLR
from tsadlib.utils.loss import EntropyLoss, GatheringLoss, harmonic_loss
from tsadlib.utils.traning_stoper import OneEarlyStopping

warnings.filterwarnings('ignore')


class BenchmarksExperiment(ExperimentBase):

    def __init__(self, args: ConfigType):
        super().__init__(args)
        self.checkpoints = os.path.join(args.checkpoints, args.model, args.dataset)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args)

        if self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=list(map(int, self.args.devices.split(','))))
        else:
            model = model.to(self.device)
        return model

    def _get_data(self, split_way: DatasetSplitEnum):
        return data_provider(self.args, split_way)

    def _select_optimizer(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)

    def _select_scheduler(self):
        args = self.args
        self.scheduler = PolynomialDecayLR(self.optimizer, warmup_steps=args.warmup_epoch * self.args.batch_size,
                                           total_steps=args.num_epochs * args.batch_size,
                                           lr=args.learning_rate, end_lr=args.end_learning_rate,
                                           power=1.0)

    def _select_criterion(self):
        self.criterion = nn.MSELoss(reduction='none')
        self.entropy_criterion = EntropyLoss()

    def _use_metric_recoder(self, setting: str):
        if self.args.use_tensorboard:
            self.writer = SummaryWriter()
        if self.args.use_wandb:
            # initialize wandb
            self.run = wandb.init(
                project="AnomalyDetectionBenchmarks",
                name=setting
            )
        self.metric_record_flag = True

    def train(self, setting: str):
        args = self.args
        device = self.device
        model = self.model

        self._use_metric_recoder(setting)

        # Load training and validating data
        train_loader, validate_loader, _ = self._get_data(DatasetSplitEnum.TRAIN_VALIDATE_SPLIT_WITH_DUPLICATES)

        if not os.path.exists(self.checkpoints):
            os.makedirs(self.checkpoints)

        early_stopping = OneEarlyStopping(patience=args.patience, root_path=self.checkpoints, file_name=setting,
                                          mode=EarlyStoppingModeEnum.MINIMIZE)

        self._select_criterion()
        criterion = self.criterion
        entropy_criterion = self.entropy_criterion

        self._select_optimizer()
        optimizer = self.optimizer
        self._select_scheduler()


        for epoch in range(args.num_epochs):
            train_losses = []

            model.train()
            epoch_time = time.time()

            for i, (x_data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1:>3} / {args.num_epochs:>3}',
                                                  ncols=100, colour='green')):
                optimizer.zero_grad()
                x_data = x_data.float().to(device)

                # Forward pass
                output_dict = model(x_data)
                output, attention = output_dict['output'], output_dict['attention']
                reconstruct_loss = criterion(output, x_data).mean()
                entropy_loss = entropy_criterion(attention)
                loss = reconstruct_loss + entropy_loss * args.hyper_parameter_lambda

                if (i + 1) % 100 == 0:
                    # record loss item
                    self._record_loss_item(loss.item())

                # Backward pass
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            validate_metric = self.validate(validate_loader, train_loader, ValidateMetricEnum.LOSS)

            train_avg_loss = np.average(train_losses)

            # Record training and validating losses
            self._record_epoch(epoch, train_avg_loss, validate_metric)

            logger.info("Epoch: {:>3} cost time: {:>10.4f}s, train loss: {:>.7f}, validate metric: {:>.7f}",
                        epoch + 1,
                        time.time() - epoch_time, train_avg_loss, validate_metric)

            # Early stopping check
            if early_stopping(float(validate_metric), model):
                logger.warning("Early stopping triggered")
                break

    def validate(self, dataloader: DataLoader, train_loader=None,
                 validate_metric: ValidateMetricEnum = ValidateMetricEnum.LOSS):
        model = self.model
        device = self.device
        model.eval()
        if validate_metric == ValidateMetricEnum.LOSS:
            losses = []
            with torch.no_grad():
                for i, (x_data, _) in enumerate(dataloader):
                    x_data = x_data.float().to(device)
                    # Forward pass
                    output_dict = model(x_data)
                    output, attention = output_dict['output'], output_dict['attention']
                    reconstruct_loss = self.criterion(output, x_data).mean()
                    entropy_loss = self.entropy_criterion(attention)
                    loss = reconstruct_loss + self.args.hyper_parameter_lambda * entropy_loss
                    losses.append(loss.item())
            return np.average(losses)
        else:
            pass

    def test(self, setting: str):
        args = self.args
        model = self.model
        device = self.device
        train_loader, test_loader = self._get_data(DatasetSplitEnum.TRAIN_NO_SPLIT)

        file_path = os.path.join(self.checkpoints, f'{setting}.pth')
        if os.path.exists(file_path):
            logger.info('Loading model weights.')
            model.load_state_dict(torch.load(file_path, map_location=self.device))
        else:
            msg = f"Model weights file {setting}.pth not found in {self.checkpoints}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Set model to evaluation mode and initialize score containers
        model.eval()
        train_scores = []
        test_scores = []
        test_labels = []
        gathering_loss = GatheringLoss(reduce=False)

        # Calculate reconstruction scores for training data
        with torch.no_grad():
            for i, (x_data, _) in enumerate(train_loader):
                x_data = x_data.float().to(device)
                output_dict = model(x_data)
                output, queries, memory = output_dict['output'], output_dict['queries'], output_dict['memory']

                # calculate loss and anomaly scores
                reconstruct_loss = self.criterion(x_data, output)
                latent_score = torch.softmax(gathering_loss(queries, memory) / args.temperature, dim=-1)
                loss = harmonic_loss(reconstruct_loss, latent_score, 'normal_mean')
                train_scores.append(loss.detach().cpu().numpy())

        train_scores = np.concatenate(train_scores, axis=0).reshape(-1)

        # Calculate reconstruction scores for test data
        for x_data, y_data in test_loader:
            x_data = x_data.float().to(device)
            output_dict = model(x_data)
            output, queries, memory = output_dict['output'], output_dict['queries'], output_dict['memory']

            # calculate loss and anomaly scores
            reconstruct_loss = self.criterion(x_data, output)
            latent_score = torch.softmax(gathering_loss(queries, memory) / args.temperature, dim=-1)
            loss = harmonic_loss(reconstruct_loss, latent_score, 'normal_mean')
            test_scores.append(loss.detach().cpu().numpy())
            test_labels.append(y_data)

        # Combine scores and labels from all batches
        test_scores = np.concatenate(test_scores, axis=0).reshape(-1)  # [total_samples, window_size]
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)  # [total_samples, window_size]

        metrics = AnomalyMetrics(test_labels, test_scores, ThresholdWayEnum.PERCENTILE, args.anomaly_ratio,
                                 train_scores)
        metrics.point_adjustment()
        result = metrics.enhanced_metrics()

        # Record evaluation metrics
        self._record_metrics(result)

        return result

    def finish(self):
        args = self.args
        if self.metric_record_flag:
            if args.use_tensorboard:
                self.writer.close()
            if args.use_wandb:
                self.run.finish()


    def _record_loss_item(self, loss_item):
        args = self.args
        if args.use_tensorboard:
            self.writer.add_scalar('loss_item', loss_item)
        if args.use_wandb:
            self.run.log({'loss_item': loss_item})

    def _record_epoch(self, epoch, train_loss, f1_score):
        args = self.args
        if args.use_tensorboard:
            self.writer.add_scalar('train_loss', train_loss)
            self.writer.add_scalar('validate_f1', f1_score)
        if args.use_wandb:
            self.run.log({
                'epoch': epoch,
                "train_loss": train_loss,
                "validate_f1": train_loss
            })

    def _record_metrics(self, result):
        args = self.args
        if args.use_tensorboard:
            self.writer.add_scalar('Metrics/Precision', result.Precision, 0)
            self.writer.add_scalar('Metrics/Recall', result.Recall, 0)
            self.writer.add_scalar('Metrics/F1_score', result.F1_score, 0)
            self.writer.add_scalar('Metrics/ROC_AUC', result.ROC_AUC, 0)

        if args.use_wandb:
            self.run.summary.update({
                "Precision": result.Precision,
                "Recall": result.Recall,
                "F1_score": result.F1_score,
                "ROC_AUC": result.ROC_AUC
            })
