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
from tsadlib import ConfigType
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

    def _get_data(self, split_way):
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
        train_loader, validate_loader, _ = self._get_data('train_validate_split')

        if not os.path.exists(args.checkpoints):
            os.makedirs(args.checkpoints)

        early_stopping = OneEarlyStopping(patience=args.patience, root_path=args.checkpoints, file_name=setting)

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

            validate_loss = self.validate(validate_loader)

            train_avg_loss = np.average(train_loss)
            validate_avg_loss = np.average(validate_loss)

            # Record training and validating losses
            self._record_epoch_losses(epoch, train_avg_loss, validate_avg_loss)

            logger.info("Epoch: {:>3} cost time: {:>10.4f}s, train loss: {:>.7f}, validate loss: {:>.7f}", epoch + 1,
                        time.time() - epoch_time, train_avg_loss, validate_avg_loss)

            # Early stopping check
            early_stopping(float(validate_avg_loss), model)
            if early_stopping.early_stop:
                logger.warning("Early stopping triggered")
                break

    def validate(self, dataloader: DataLoader):
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()
        losses = []
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(dataloader):
                batch_x = batch_x.float().to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                losses.append(loss.item())
        return losses

    def test(self, setting: str):
        args = self.args
        model = self.model
        device = self.device
        train_loader, test_loader = self._get_data(split_way='train_no_split')

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

        metrics = AnomalyMetrics(test_labels, test_scores, 'percentile', args.anomaly_ratio, train_scores)
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

    def _record_epoch_losses(self, epoch, train_loss, validate_loss):
        args = self.args
        if args.use_tensorboard:
            self.writer.add_scalar('Loss/train', train_loss)
            self.writer.add_scalar('Loss/validate', validate_loss)
        if args.use_wandb:
            self.run.log({
                'epoch': epoch,
                "train_loss": train_loss,
                "validate_loss": validate_loss
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
