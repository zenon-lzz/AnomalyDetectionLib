import os.path
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from tsadlib import logger
from tsadlib.preprocess import MBADataset, NABDataset, MSDSDataset, MSLSMAPDataset, SMDDataset, WADIDataset, UCRDataset
from tsadlib.utils.constants import PROJECT_ROOT

DATASET_ROOT = '/Users/liuzhenzhou/Documents/backup/datasets/anomaly_detection'

class Dataset_Preprocess(unittest.TestCase):

    # Test methods must begin with test_
    def test_mba_preprocess(self):
        mba = MBADataset(f'{DATASET_ROOT}/raw/MBA',
                         f'{DATASET_ROOT}/npy/MBA')
        mba.load_data()
        mba.preprocess()
        mba.save()

        path = os.path.join(PROJECT_ROOT, 'Results/Plots')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        pdf = PdfPages(os.path.join(path, 'MBA.pdf'))
        figures = mba.visualize()
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
        pdf.close()
        logger.info(f'MBA\'s statistics:\n{pd.DataFrame([mba.get_statistics()]).to_string(index=False)}')

    # Test methods must begin with test_
    def test_msds_preprocess(self):
        msds = MSDSDataset(f'{DATASET_ROOT}/raw/MSDS',
                           f'{DATASET_ROOT}/npy/MSDS')
        msds.load_data()
        msds.preprocess()
        msds.save()

        path = os.path.join(PROJECT_ROOT, 'Results/Plots')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        pdf = PdfPages(os.path.join(path, 'MSDS.pdf'))
        figures = msds.visualize()
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
        pdf.close()

        logger.info(f'MSDS\'s statistics:\n{pd.DataFrame([msds.get_statistics()]).to_string(index=False)}')

    # Test methods must begin with test_
    def test_nab_preprocess(self):
        nab = NABDataset(f'{DATASET_ROOT}/raw/NAB',
                         f'{DATASET_ROOT}/npy/NAB')
        nab.load_data()
        nab.preprocess()
        nab.save()

        path = os.path.join(PROJECT_ROOT, 'Results/Plots')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        pdf = PdfPages(os.path.join(path, 'NAB.pdf'))
        figures = nab.visualize()
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
        pdf.close()

        logger.info(f'NAB\'s statistics:\n{pd.DataFrame(nab.get_statistics()).to_string(index=False)}')

    def test_msl_smap_preprocess(self):
        # Test MSL dataset
        msl = MSLSMAPDataset(
            f'{DATASET_ROOT}/raw/SMAP_MSL',
            f'{DATASET_ROOT}/npy/MSL',
            spacecraft='MSL'
        )
        msl.load_data()
        msl.preprocess()
        msl.save()

        path = os.path.join(PROJECT_ROOT, 'Results/Plots')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        pdf = PdfPages(os.path.join(path, 'MSL.pdf'))
        figures = msl.visualize()
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
        pdf.close()

        logger.info(f'MSL\'s statistics:\n{pd.DataFrame(msl.get_statistics()).to_string(index=False)}')

        # Test SMAP dataset
        smap = MSLSMAPDataset(
            f'{DATASET_ROOT}/raw/SMAP_MSL',
            f'{DATASET_ROOT}/npy/SMAP',
            spacecraft='SMAP'
        )
        smap.load_data()
        smap.preprocess()
        smap.save()

        pdf = PdfPages(os.path.join(path, 'SMAP.pdf'))
        figures = smap.visualize()
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
        pdf.close()

        logger.info(f'SMAP\'s statistics:\n{pd.DataFrame(smap.get_statistics()).to_string(index=False)}')

    def test_smd_preprocess(self):
        smd = SMDDataset(f'{DATASET_ROOT}/raw/SMD',
                         f'{DATASET_ROOT}/npy/SMD')
        smd.load_data()
        smd.preprocess()
        smd.save()

        path = os.path.join(PROJECT_ROOT, 'Results/Plots')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        pdf = PdfPages(os.path.join(path, 'SMD.pdf'))
        figures = smd.visualize()
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
        pdf.close()

        logger.info(f'SMD\'s statistics:\n{pd.DataFrame(smd.get_statistics()).to_string(index=False)}')

    def test_wadi_preprocess(self):
        wadi = WADIDataset(f'{DATASET_ROOT}/raw/WADI',
                           f'{DATASET_ROOT}/npy/WADI')
        wadi.load_data()
        wadi.preprocess()
        wadi.save()

        path = os.path.join(PROJECT_ROOT, 'Results/Plots')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        pdf = PdfPages(os.path.join(path, 'WADI.pdf'))
        figures = wadi.visualize()
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
        pdf.close()

        logger.info(f'WADI\'s statistics:\n{pd.DataFrame([wadi.get_statistics()]).to_string(index=False)}')

    def test_ucr_preprocess(self):
        ucr = UCRDataset(f'{DATASET_ROOT}/raw/UCR',
                         f'{DATASET_ROOT}/npy/UCR')
        ucr.load_data()
        ucr.preprocess()
        ucr.save()

        path = os.path.join(PROJECT_ROOT, 'Results/Plots')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        pdf = PdfPages(os.path.join(path, 'UCR.pdf'))
        figures = ucr.visualize()
        for fig in figures:
            pdf.savefig(fig)
        plt.close()
        pdf.close()

        logger.info(f'UCR\'s statistics:\n{pd.DataFrame(ucr.get_statistics()).to_string(index=False)}')


if __name__ == '__main__':
    unittest.main()
