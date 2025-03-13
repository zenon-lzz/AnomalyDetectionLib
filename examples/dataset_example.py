"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-11
@Description：数据集处理模块使用示例
==================================================
"""

import os

import numpy as np
import pandas as pd
from loguru import logger

# 导入数据集处理模块
from tsadlib import MBADataset, SMAPDataset, SWaTDataset


def smap_dataset_example():
    """SMAP数据集处理示例"""
    logger.info("=== SMAP数据集处理示例 ===")

    # 假设数据存储在 data/SMAP 目录下
    data_dir = os.path.join("data", "SMAP")
    save_dir = os.path.join("data", "SMAP", "processed")

    # 创建SMAP数据集处理对象
    smap_dataset = SMAPDataset(
        data_dir=data_dir,
        save_dir=save_dir,
        train_ratio=0.7,
        valid_ratio=0.1,
        test_ratio=0.2,
        normalize=True,
        normalize_method='minmax'
    )

    try:
        # 加载数据
        smap_dataset.load_data()

        # 预处理数据
        smap_dataset.preprocess()

        # 分割数据集
        train_data, valid_data, test_data = smap_dataset.split_data()

        # 准备滑动窗口数据（适用于深度学习模型）
        window_size = 100
        stride = 10
        train_windows, valid_windows, test_windows = smap_dataset.prepare_sliding_window_data(
            window_size=window_size, stride=stride)

        # 获取数据集统计信息
        stats = smap_dataset.get_statistics()
        logger.info(f"数据集统计信息: {stats}")

        # 可视化数据
        smap_dataset.visualize(channel_index=0, time_range=(0, 1000))

        # 保存处理后的数据
        smap_dataset.save(format='npy')

        logger.info("SMAP数据集处理完成")

    except FileNotFoundError as e:
        logger.error(f"数据文件不存在: {e}")
        logger.info("请先下载SMAP数据集并放置到正确目录")


def swat_dataset_example():
    """SWaT数据集处理示例"""
    logger.info("=== SWaT数据集处理示例 ===")

    # 假设数据存储在 data/SWaT 目录下
    data_dir = os.path.join("data", "SWaT")
    save_dir = os.path.join("data", "SWaT", "processed")

    # 创建SWaT数据集处理对象
    swat_dataset = SWaTDataset(
        data_dir=data_dir,
        save_dir=save_dir,
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        normalize=True,
        normalize_method='standard',
        resample_freq='1min'  # 重采样频率
    )

    try:
        # 加载数据
        swat_dataset.load_data()

        # 预处理数据
        swat_dataset.preprocess()

        # 分割数据集
        train_data, valid_data, test_data = swat_dataset.split_data()

        # 获取特征名称
        feature_names = swat_dataset.get_feature_names()
        logger.info(f"特征名称: {feature_names[:5]}...")  # 只显示前5个

        # 获取数据集统计信息
        stats = swat_dataset.get_statistics()
        logger.info(f"数据集统计信息摘要: 训练样本数 {stats['train_samples']}, 测试样本数 {stats['test_samples']}")

        # 可视化数据
        swat_dataset.visualize(features=feature_names[:3])  # 显示前3个特征

        # 保存处理后的数据
        swat_dataset.save(format='pkl')

        logger.info("SWaT数据集处理完成")

    except FileNotFoundError as e:
        logger.error(f"数据文件不存在: {e}")
        logger.info("请先下载SWaT数据集并放置到正确目录")


def mba_dataset_example():
    """MBA数据集处理示例"""
    logger.info("=== MBA数据集处理示例 ===")

    # 假设数据存储在 data/MBA 目录下
    data_dir = os.path.join("data", "MBA")
    save_dir = os.path.join("data", "MBA", "processed")

    # 创建MBA数据集处理对象
    mba_dataset = MBADataset(
        data_dir=data_dir,
        save_dir=save_dir,
        train_ratio=0.6,
        valid_ratio=0.2,
        test_ratio=0.2,
        normalize=True,
        normalize_method='minmax'
    )

    try:
        # 加载数据
        mba_dataset.load_data()

        # 预处理数据
        mba_dataset.preprocess()

        # 分割数据集
        train_data, valid_data, test_data = mba_dataset.split_data()

        # 准备滑动窗口数据，并展平窗口
        window_size = 50
        stride = 5
        train_windows, valid_windows, test_windows = mba_dataset.prepare_sliding_window_data(
            window_size=window_size, stride=stride, flatten=True)

        # 获取数据集统计信息
        stats = mba_dataset.get_statistics()
        logger.info(f"数据集统计信息: 总样本数 {stats['total_samples']}, 传感器数量 {stats['num_sensors']}")

        # 可视化数据
        mba_dataset.visualize(sensor_indices=[0, 1, 2], time_range=(0, 1000))  # 显示前3个传感器的前1000个点

        # 保存处理后的数据
        mba_dataset.save(format='csv')

        logger.info("MBA数据集处理完成")

    except FileNotFoundError as e:
        logger.error(f"数据文件不存在: {e}")
        logger.info("请先下载MBA数据集并放置到正确目录")


def create_dummy_data():
    """创建示例数据（用于演示）"""
    logger.info("创建示例数据...")

    # 创建目录
    for dataset in ['SMAP', 'SWaT', 'MBA']:
        os.makedirs(os.path.join("data", dataset), exist_ok=True)

    # 创建SMAP示例数据
    smap_data_dir = os.path.join("data", "SMAP")
    n_samples = 5000
    n_channels = 10

    # 创建训练和测试数据
    train_data = np.random.randn(n_samples, n_channels)
    test_data = np.random.randn(n_samples // 2, n_channels)

    # 创建标签（大部分为0，少量为1）
    labels = np.zeros((n_samples + n_samples // 2, n_channels))
    anomaly_indices = np.random.choice(n_samples + n_samples // 2, size=int((n_samples + n_samples // 2) * 0.05),
                                       replace=False)
    labels[anomaly_indices] = 1

    # 保存数据
    np.save(os.path.join(smap_data_dir, "train.npy"), train_data)
    np.save(os.path.join(smap_data_dir, "test.npy"), test_data)
    np.save(os.path.join(smap_data_dir, "labels.npy"), labels)

    # 创建通道信息
    channel_info = {str(i): {"name": f"Channel_{i}", "index": i} for i in range(n_channels)}
    with open(os.path.join(smap_data_dir, "channel_info.json"), 'w') as f:
        import json
        json.dump(channel_info, f, indent=2)

    # 创建SWaT示例数据
    swat_data_dir = os.path.join("data", "SWaT")
    n_samples = 5000
    n_features = 20

    # 创建时间戳
    start_date = pd.Timestamp('2015-12-22')
    timestamps = [start_date + pd.Timedelta(minutes=i) for i in range(n_samples)]

    # 创建特征名称
    feature_names = [f"Sensor_{i}" for i in range(n_features)]

    # 创建训练数据（正常）
    train_data = np.random.randn(n_samples, n_features)
    train_df = pd.DataFrame(train_data, columns=feature_names)
    train_df['Timestamp'] = timestamps
    train_df['Normal/Attack'] = 0  # 全部正常

    # 创建测试数据（包含异常）
    test_data = np.random.randn(n_samples // 2, n_features)
    test_timestamps = [start_date + pd.Timedelta(minutes=n_samples + i) for i in range(n_samples // 2)]
    test_df = pd.DataFrame(test_data, columns=feature_names)
    test_df['Timestamp'] = test_timestamps
    test_df['Normal/Attack'] = 0  # 默认正常

    # 添加一些异常
    anomaly_indices = np.random.choice(n_samples // 2, size=int((n_samples // 2) * 0.1), replace=False)
    test_df.loc[anomaly_indices, 'Normal/Attack'] = 1

    # 保存数据
    train_df.to_csv(os.path.join(swat_data_dir, "SWaT_Dataset_Normal_v1.csv"), index=False)
    test_df.to_csv(os.path.join(swat_data_dir, "SWaT_Dataset_Attack_v0.csv"), index=False)

    # 创建MBA示例数据
    mba_data_dir = os.path.join("data", "MBA")
    n_samples = 5000
    n_sensors = 15

    # 创建传感器名称
    sensor_names = [f"Sensor_{i}" for i in range(n_sensors)]

    # 创建数据
    data = np.random.randn(n_samples, n_sensors)
    timestamps = [start_date + pd.Timedelta(minutes=i) for i in range(n_samples)]

    # 创建DataFrame
    df = pd.DataFrame(data, columns=sensor_names)
    df['Timestamp'] = timestamps

    # 创建标签
    labels = np.zeros(n_samples)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    labels[anomaly_indices] = 1
    labels_df = pd.DataFrame({'Label': labels})

    # 保存数据
    df.to_csv(os.path.join(mba_data_dir, "mba_data.csv"), index=False)
    labels_df.to_csv(os.path.join(mba_data_dir, "mba_labels.csv"), index=False)

    # 创建传感器信息
    sensor_info = {str(i): {"name": f"Sensor_{i}", "description": f"Description for sensor {i}"} for i in
                   range(n_sensors)}
    with open(os.path.join(mba_data_dir, "sensor_info.json"), 'w') as f:
        import json
        json.dump(sensor_info, f, indent=2)

    logger.info("示例数据创建完成")


if __name__ == "__main__":
    # 创建示例数据（首次运行时打开此注释）
    # create_dummy_data()

    # 运行SMAP数据集示例
    smap_dataset_example()

    # 运行SWaT数据集示例
    swat_dataset_example()

    # 运行MBA数据集示例
    mba_dataset_example()
