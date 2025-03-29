"""
=================================================
@Author: Zenon
@Date: 2025-03-26
@Description：Debugger File
==================================================
"""



if __name__ == '__main__':
    import os
    import time

    import numpy as np
    import torch
    from sklearn.metrics import precision_recall_fscore_support
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from tsadlib.configs.constants import PROJECT_ROOT
    from tsadlib.configs.type import ConfigType
    from tsadlib.data_provider.data_factory import data_provider
    from tsadlib.metrics.threshold import percentile_threshold
    from tsadlib.models.MEMTO import MEMTO
    from tsadlib.utils.adjustment import point_adjustment
    from tsadlib.utils.clustering import k_means_clustering
    from tsadlib.utils.logger import logger
    from tsadlib.utils.loss import EntropyLoss, GatheringLoss
    from tsadlib.utils.traning_stoper import OneEarlyStopping

    # Set up device for computation (CUDA GPU, Apple M1/M2 GPU, or CPU)
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    logger.info(f'use device: {device}')
    device = torch.device(device)

    # Define paths for dataset and model checkpoints
    # DATASET_ROOT = 'E:\\liuzhenzhou\\datasets'
    # DATASET_ROOT = '/Users/liuzhenzhou/Documents/backup/datasets/anomaly_detection/npy'
    DATASET_ROOT = '/home/lzz/Desktop/datasets'
    DATASET_TYPE = 'MSL'  # Mars Science Laboratory dataset
    MODEL = 'MEMTO'
    CHECKPOINTS = os.path.join(PROJECT_ROOT, 'checkpoints', MODEL)

    writer = SummaryWriter(os.path.join(PROJECT_ROOT, 'runs', MODEL).__str__())

    # Configure TimesNet hyperparameters and training settings
    args = ConfigType(**{
        'model': MODEL,
        'mode': 'train',
        'dataset_root_path': os.path.join(DATASET_ROOT, DATASET_TYPE),
        'window_size': 100,
        'batch_size': 256,
        'd_model': 8,
        'dimension_fcl': 16,
        'encoder_layers': 3,
        'input_channels': 55,
        'output_channels': 55,
        'num_memory': 10,
        'hyper_parameter_lambda': 0.01,
        'dropout': 0.1,
        'anomaly_ratio': 1,
        'num_epochs': 100,
        'learning_rate': 1e-4
    })

    # Load training and testing data
    train_loader, validate_loader, test_loader, k_loader = data_provider(args, split_way='train_validate_k_split')

    # Initialize model and training components
    model = MEMTO(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    entropy_criterion = EntropyLoss()
    early_stopping = OneEarlyStopping(args.patience, CHECKPOINTS, DATASET_TYPE)
    train_steps = len(train_loader)
    logger.info('The first phase training starts')

    for epoch in range(args.num_epochs):
        model.train()
        train_losses = []
        reconstruct_losses = []
        entropy_losses = []
        validate_losses = []
        iter_count = 0
        epoch_time = time.time()

        model.train()
        for i, (x_data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} / {args.num_epochs}')):
            iter_count += 1
            optimizer.zero_grad()
            x_data = x_data.float().to(device)

            # Forward pass
            output_dict = model(x_data)
            output, attention = output_dict['output'], output_dict['attention']
            reconstruct_loss = criterion(output, x_data)
            entropy_loss = entropy_criterion(attention)
            loss = reconstruct_loss + args.hyper_parameter_lambda * entropy_loss

            if (i + 1) % 100 == 0:
                writer.add_scalar('Loss/Train', loss.item(), epoch * train_steps + i)
                writer.add_scalar('Loss/Reconstruct', reconstruct_loss.item(), epoch * train_steps + i)
                writer.add_scalar('Loss/Entropy', entropy_loss.item(), epoch * train_steps + i)

            # Backward pass
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            reconstruct_losses.append(reconstruct_loss.item())
            entropy_losses.append(entropy_loss)

        model.eval()
        with torch.no_grad():
            for i, (x_data, _) in enumerate(validate_loader):
                x_data = x_data.float().to(device)
                # Forward pass
                output_dict = model(x_data)
                output, attention = output_dict['output'], output_dict['attention']
                reconstruct_loss = criterion(output, x_data)
                entropy_loss = entropy_criterion(attention)
                loss = reconstruct_loss + args.hyper_parameter_lambda * entropy_loss
                validate_losses.append(loss.item())

        train_avg_loss = np.average(train_losses)
        validate_avg_loss = np.average(validate_losses)
        logger.info("Epoch: {:>2} cost time: {:<10.4f}s, train loss: {:<.7f}, validate loss: {:<.7f}", epoch + 1,
                    time.time() - epoch_time, train_avg_loss, validate_avg_loss)

        writer.add_scalars("Loss", {"Train": train_avg_loss, "Validation": validate_avg_loss}, epoch)

        # Early stopping check
        early_stopping(validate_avg_loss, model)
        if early_stopping.early_stop:
            logger.warning("Early stopping triggered")
            break

    model.eval()
    outputs = []

    # sample 10% of training data to generate queries
    with torch.no_grad():
        for i, (x_data, _) in enumerate(k_loader):
            x_data = x_data.float().to(device)
            # Forward pass
            outputs.append(model(x_data)['queries'])

    # Apply K-means clustering algorithm to cluster the queries and designate each centroid as initial value of a memory item.
    outputs = torch.cat(outputs, dim=0)
    memory_init_embedding = k_means_clustering(outputs, args.num_memory, args.d_model)

    model = MEMTO(args, memory_init_embedding.detach()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stopping = OneEarlyStopping(args.patience, CHECKPOINTS, DATASET_TYPE)

    logger.info('The second phase training starts')
    for epoch in range(args.num_epochs):
        model.train()
        train_losses = []
        reconstruct_losses = []
        entropy_losses = []
        validate_losses = []
        iter_count = 0
        epoch_time = time.time()

        model.train()
        for i, (x_data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} / {args.num_epochs}')):
            iter_count += 1
            optimizer.zero_grad()
            x_data = x_data.float().to(device)

            # Forward pass
            output_dict = model(x_data)
            output, attention = output_dict['output'], output_dict['attention']
            reconstruct_loss = criterion(output, x_data)
            entropy_loss = entropy_criterion(attention)
            loss = reconstruct_loss + args.hyper_parameter_lambda * entropy_loss

            if (i + 1) % 100 == 0:
                writer.add_scalar('Loss/Train', loss.item(), epoch * train_steps + i)
                writer.add_scalar('Loss/Reconstruct', reconstruct_loss.item(), epoch * train_steps + i)
                writer.add_scalar('Loss/Entropy', entropy_loss.item(), epoch * train_steps + i)

            # Backward pass
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            reconstruct_losses.append(reconstruct_loss.item())
            entropy_losses.append(entropy_loss)

        model.eval()
        with torch.no_grad():
            for i, (x_data, _) in enumerate(validate_loader):
                x_data = x_data.float().to(device)
                # Forward pass
                output_dict = model(x_data)
                output, attention = output_dict['output'], output_dict['attention']
                reconstruct_loss = criterion(output, x_data)
                entropy_loss = entropy_criterion(attention)
                loss = reconstruct_loss + args.hyper_parameter_lambda * entropy_loss
                validate_losses.append(loss.item())

        train_avg_loss = np.average(train_losses)
        validate_avg_loss = np.average(validate_losses)
        logger.info("Epoch: {:>2} cost time: {:<10.4f}s, train loss: {:<.7f}, validate loss: {:<.7f}", epoch + 1,
                    time.time() - epoch_time, train_avg_loss, validate_avg_loss)

        writer.add_scalars("Loss", {"Train": train_avg_loss, "Validation": validate_avg_loss}, epoch)

        # Early stopping check
        early_stopping(validate_avg_loss, model)
        if early_stopping.early_stop:
            logger.warning("Early stopping triggered")
            break

    train_attention_energy = []
    validate_attention_energy = []
    test_attention_energy = []
    test_labels = []
    criterion = nn.MSELoss(reduction='none')
    gathering_loss = GatheringLoss(reduce=False)
    temperature = args.temperature

    logger.info('Test Phase Starts')

    model.eval()
    # Calculate Anomaly Scores in training set.
    for i, (x_data, _) in enumerate(train_loader):
        x_data = x_data.float().to(device)

        output_dict = model(x_data)
        output, queries, memory = output_dict['output'], output_dict['queries'], output_dict['memory']

        # calculate loss and anomaly scores
        reconstruct_loss = torch.mean(criterion(x_data, output), dim=-1)
        latent_score = torch.softmax(gathering_loss(queries, memory) / temperature, dim=-1)
        loss = latent_score * reconstruct_loss

        train_attention_energy.append(loss.detach().cpu().numpy())

    train_attention_energy = np.concatenate(train_attention_energy, axis=0).reshape(-1)
    # Calculate Anomaly Scores in validation set.
    for i, (x_data, _) in enumerate(validate_loader):
        x_data = x_data.float().to(device)

        output_dict = model(x_data)
        output, queries, memory = output_dict['output'], output_dict['queries'], output_dict['memory']

        # calculate loss and anomaly scores
        reconstruct_loss = torch.mean(criterion(x_data, output), dim=-1)
        latent_score = torch.softmax(gathering_loss(queries, memory) / temperature, dim=-1)
        loss = latent_score * reconstruct_loss

        validate_attention_energy.append(loss.detach().cpu().numpy())

    validate_attention_energy = np.concatenate(validate_attention_energy, axis=0).reshape(-1)
    combined_energy = np.concatenate([train_attention_energy, validate_attention_energy], axis=0)
    threshold = percentile_threshold(combined_energy, 100 - args.anomaly_ratio)
    logger.info('Threshold is {:.4f}', threshold)

    # Calculate reconstruction scores for test data
    for i, (x_data, labels) in enumerate(test_loader):
        x_data = x_data.float().to(device)
        output_dict = model(x_data)
        output, queries, memory = output_dict['output'], output_dict['queries'], output_dict['memory']

        reconstruct_loss = torch.mean(criterion(x_data, output), dim=-1)
        latent_score = torch.softmax(gathering_loss(queries, memory) / temperature, dim=-1)
        loss = latent_score * reconstruct_loss
        test_attention_energy.append(loss.detach().cpu().numpy())
        test_labels.append(labels)

    # Combine scores and labels from all batches
    test_energy = np.concatenate(test_attention_energy, axis=0).reshape(-1)  # [total_samples, window_size]
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)  # [total_samples, window_size]

    # Generate predictions based on threshold
    pred_labels = (test_energy > threshold).astype(int)
    gt_labels = test_labels.astype(int)

    # Calculate evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, pred_labels, average='binary')
    logger.success('Before point-adjustment:\nPrecision: {:.2f}\nRecall: {:.4f}\nF1-score: {:.2f}', precision, recall,
                   f1)

    # Apply point-adjustment strategy
    gt, pred = point_adjustment(test_labels, pred_labels)

    # Calculate evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(gt, pred, average='binary')
    logger.success('After point-adjustment:\nPrecision: {:.2f}\nRecall: {:.4f}\nF1-score: {:.2f}', precision, recall,
                   f1)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    plt.plot(test_energy)
    plt.axhline(y=threshold, color='r', linestyle='--', label='threshold')
    anomaly_indices = np.where(gt == 1)[0]
    plt.plot(np.arange(len(test_energy))[anomaly_indices], test_energy[anomaly_indices], 'r.', markersize=2,
             label='Anomaly')
    plt.title('MEMTO Model Evaluation')
    plt.xlabel('TimeStamp')
    plt.ylabel('Anomaly Scores')
    plt.legend()