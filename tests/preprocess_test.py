import os.path
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from tsadlib import logger
from tsadlib.preprocess import MBADataset
from tsadlib.utils.constants import PROJECT_ROOT


class Dataset_Preprocess(unittest.TestCase):

    # Test methods must begin with test_
    def test_mba_preprocess(self):
        mba = MBADataset("/Users/liuzhenzhou/Documents/backup/datasets/anomaly_detection/raw/MBA",
                         '/Users/liuzhenzhou/Documents/backup/datasets/anomaly_detection/npy/MBA')
        mba.load_data()
        mba.preprocess()
        mba.save()
        logger.info(f'MBA\'s statistics:\n{pd.DataFrame([mba.get_statistics()]).to_string(index=False)}')

    def test_mba_visualize(self):
        mba = MBADataset("/Users/liuzhenzhou/Documents/backup/datasets/anomaly_detection/raw/MBA",
                         '/Users/liuzhenzhou/Documents/backup/datasets/anomaly_detection/npy/MBA')
        mba.load_data()

        path = os.path.join(PROJECT_ROOT, 'Results/Plots')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        pdf = PdfPages(os.path.join(path, 'MBA.pdf'))
        figures = mba.visualize()
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
        pdf.close()


if __name__ == '__main__':
    unittest.main()
