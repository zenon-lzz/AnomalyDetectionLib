"""
=================================================
@Author: Zenon
@Date: 2025-05-01
@Description：
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
from tsadlib.utils.traning_stoper import OneEarlyStopping

warnings.filterwarnings('ignore')


class BenchmarksExperiment(ExperimentBase):

    def __init__(self, args: ConfigType):
        super().__init__(args)
        args.checkpoints = os.path.join(args.checkpoints, args.model, args.dataset)

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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        self.criterion = nn.MSELoss()

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
        train_loader, validate_loader = self._get_data(DatasetSplitEnum.TRAIN_NO_SPLIT)

        if not os.path.exists(args.checkpoints):
            os.makedirs(args.checkpoints)

        early_stopping = OneEarlyStopping(patience=args.patience, root_path=args.checkpoints, file_name=setting,
                                          mode=EarlyStoppingModeEnum.MAXIMIZE)

        optimizer = self._select_optimizer()
        self._select_criterion()
        criterion = self.criterion

        for epoch in range(args.num_epochs):
            train_loss = []

            model.train()
            epoch_time = time.time()

            for i, (batch_x, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1:>3} / {args.num_epochs:>3}',
                                                  ncols=100, colour='green')):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(device)

                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)

                if (i + 1) % 100 == 0:
                    # record loss item
                    self._record_loss_item(loss.item())

                # Backward pass
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            f1_score = self.validate(validate_loader, train_loader, ValidateMetricEnum.F1_SCORE)

            train_avg_loss = np.average(train_loss)

            # Record training and validating losses
            self._record_epoch(epoch, train_avg_loss, f1_score)

            logger.info("Epoch: {:>3} cost time: {:>10.4f}s, train loss: {:>.7f}, validate f1-score: {:>.7f}",
                        epoch + 1,
                        time.time() - epoch_time, train_avg_loss, f1_score)

            # Early stopping check
            if early_stopping(float(f1_score), model):
                logger.warning("Early stopping triggered")
                break

    def validate(self, dataloader: DataLoader, train_loader=None,
                 validate_metric: ValidateMetricEnum = ValidateMetricEnum.LOSS):
        model = self.model
        device = self.device
        model.eval()
        if validate_metric == ValidateMetricEnum.LOSS:
            losses = []
            criterion = self.criterion
            with torch.no_grad():
                for i, (batch_x, _) in enumerate(dataloader):
                    batch_x = batch_x.float().to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                    losses.append(loss.item())
            return np.average(losses)
        else:
            train_scores = []
            test_scores = []
            test_labels = []
            anomaly_criterion = nn.MSELoss(reduction='none')

            # Calculate reconstruction scores for training data
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    batch_x = batch_x.float().to(device)
                    # reconstruction
                    outputs = model(batch_x)
                    # criterion
                    score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
                    score = score.detach().cpu().numpy()
                    train_scores.append(score)

            train_scores = np.concatenate(train_scores, axis=0).reshape(-1)

            # Calculate reconstruction scores for test data
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.float().to(device)
                outputs = model(batch_x)

                # Calculate reconstruction error as anomaly score
                score = torch.mean(anomaly_criterion(outputs, batch_x), dim=-1)
                test_scores.append(score.detach().cpu().numpy())
                test_labels.append(batch_y)

            # Combine scores and labels from all batches
            test_scores = np.concatenate(test_scores, axis=0).reshape(-1)  # [total_samples, window_size]
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)  # [total_samples, window_size]

            metrics = AnomalyMetrics(test_labels, test_scores, ThresholdWayEnum.PERCENTILE, self.args.anomaly_ratio,
                                     train_scores)
            # metrics.point_adjustment()
            return metrics.f1_score()

    def test(self, setting: str):
        args = self.args
        model = self.model
        device = self.device
        train_loader, test_loader = self._get_data(DatasetSplitEnum.TRAIN_NO_SPLIT)

        file_path = os.path.join(args.checkpoints, f'{setting}.pth')
        if os.path.exists(file_path):
            logger.info('Loading model weights.')
            model.load_state_dict(torch.load(file_path, map_location=self.device))
        else:
            msg = f"Model weights file {setting}.pth not found in {args.checkpoints}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Set model to evaluation mode and initialize score containers
        model.eval()
        train_scores = []
        test_scores = []
        test_labels = []
        anomaly_criterion = nn.MSELoss(reduction='none')

        # Calculate reconstruction scores for training data
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(device)
                # reconstruction
                outputs = model(batch_x)
                # criterion
                score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                train_scores.append(score)

        train_scores = np.concatenate(train_scores, axis=0).reshape(-1)

        # Calculate reconstruction scores for test data
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x)

            # Calculate reconstruction error as anomaly score
            score = torch.mean(anomaly_criterion(outputs, batch_x), dim=-1)
            test_scores.append(score.detach().cpu().numpy())
            test_labels.append(batch_y)

        # Combine scores and labels from all batches
        test_scores = np.concatenate(test_scores, axis=0).reshape(-1)  # [total_samples, window_size]
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)  # [total_samples, window_size]

        metrics = AnomalyMetrics(test_labels, test_scores, ThresholdWayEnum.PERCENTILE, args.anomaly_ratio,
                                 train_scores)
        metrics.point_adjustment()
        result = metrics.common_metrics()

        # Record evaluation metrics
        self._record_metrics(result)

        logger.success('Result:\nPrecision: {:.2f}\nRecall: {:.2f}\nF1-score: {:.2f}',
                       result.Precision,
                       result.Recall, result.F1_score)

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
