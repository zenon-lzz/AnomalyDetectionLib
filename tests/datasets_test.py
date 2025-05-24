"""
=================================================
@Author: Zenon
@Date: 2025-05-24
@Description：Anomaly ratio calculate for various datasets.
==================================================
"""
import os
import unittest

import numpy as np


class AnomalyRatio(unittest.TestCase):
    datasets_root_directory = '/home/lzz/Desktop/datasets/ad'

    def test_MSL(self):
        root_path = os.path.join(self.datasets_root_directory, 'MSL')
        train_data = np.load(os.path.join(root_path, 'MSL_train.npy'))
        test_data = np.load(os.path.join(root_path, 'MSL_test.npy'))
        labels = np.load(os.path.join(root_path, 'MSL_test_label.npy'))
        print(f'Training set shape: {train_data.shape}')
        print(f'Test set shape: {test_data.shape}')
        print(f'Labels shape: {labels.shape}')
        anomaly_ratio = np.sum(labels) / (train_data.shape[0] + test_data.shape[0])
        print(f'Anomaly ratio: {anomaly_ratio}')

        """
        Training set shape: (58317, 55)
        Test set shape: (73729, 55)
        Labels shape: (73729,)
        Anomaly ratio: 0.05881283795041122
        """
