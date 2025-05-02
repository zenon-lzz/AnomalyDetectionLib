"""
=================================================
@Author: Zenon
@Date: 2025-05-01
@Description：Python Grammar Learning
==================================================
"""
import unittest

from tsadlib import Metric


class GrammarLearning(unittest.TestCase):

    def test_basic(self):
        print(Metric(**{
            'Precision': 1,
            'Recall': 2,
            'F1_score': 2,
            'ROC_AUC': 4
        }))
