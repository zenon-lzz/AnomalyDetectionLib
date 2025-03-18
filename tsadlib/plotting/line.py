"""
=================================================
@Author: Zenon
@Date: 2025-03-13
@Description: Line plot utilities for time series visualization
Contains functions for smoothing signals and plotting multi-dimensional time series data
==================================================
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def smooth(y, box_pts=1):
    """Smooth a 1D signal using a moving average filter.

    Args:
        y (np.ndarray): Input signal to be smoothed
        box_pts (int, optional): Size of the moving average window. Defaults to 1.

    Returns:
        np.ndarray: Smoothed signal using moving average filtering
    """
    # Create normalized box filter kernel
    box = np.ones(box_pts) / box_pts

    # Apply convolution with 'same' mode to maintain input signal length
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


class LinePlot:
    """Class containing static methods for time series line plotting."""

    @staticmethod
    def plot_time_series(y_data: np.ndarray,
                         column_names: List[str],
                         x_data: np.ndarray | None = None,
                         labels_data: np.ndarray | None = None,
                         dims: int | list[int] | None = None,
                         title: str | None = None) -> Figure:
        """Plot multidimensional time series data with optional anomaly highlighting.

        Args:
            y_data (np.ndarray): 2D array of time series data (samples x dimensions)
            column_names (List[str]): Names for each dimension/column
            x_data (np.ndarray | None, optional): Custom x-axis values. Defaults to None.
            labels_data (np.ndarray | None, optional): Binary labels for anomaly highlighting. Defaults to None.
            dims (int | list[int] | None, optional): Specific dimensions to plot. Defaults to None (all dimensions).
            title (str | None, optional): Title for the figure. Defaults to None.

        Returns:
            Figure: Matplotlib figure containing the plotted time series

        Raises:
            ValueError: If y_data is not 2D or dims has invalid type
        """
        # Input validation
        if len(y_data.shape) != 2:
            raise ValueError("y_data must be two-dimensional")
        L, D = y_data.shape

        # Filter data based on provided x_data range
        if x_data is not None:
            start_time = x_data[0]
            end_time = x_data[-1]
            mask = np.arange(L)

            # Create mask for the time range
            mask = (mask >= start_time) & (mask <= end_time)

            # Filter data based on the mask
            y_data = y_data[mask]
            if labels_data is not None:
                labels_data = labels_data[mask]

            # Update L after filtering
            L = len(y_data)

        # Determine which dimensions to plot
        if dims is None:
            dims_to_plot = range(D)
        elif isinstance(dims, int):
            dims_to_plot = [dims]
        elif isinstance(dims, list):
            dims_to_plot = dims
        else:
            raise ValueError('dims must be integer, list or None')

        # Create subplots
        num_dims = len(dims_to_plot)
        # Calculate figure height based on number of dimensions
        min_height_per_subplot = 2.0  # Minimum height for each subplot in inches
        figure_height = max(num_dims * min_height_per_subplot, 6.0)  # Ensure minimum total height

        # Create figure with calculated dimensions
        figure, axes = plt.subplots(num_dims, 1,
                                    figsize=(10, figure_height),  # Width fixed at 10 inches
                                    sharex=True)

        # Adjust subplot parameters for better spacing
        plt.subplots_adjust(
            top=0.90,  # Reduced top margin to leave space for main title
            bottom=0.05,  # Bottom margin
            hspace=0.4,  # Height space between subplots
            left=0.1,  # Left margin
            right=0.95  # Right margin
        )

        if num_dims == 1:
            axes = [axes]

        # Add title with adjusted position
        if title:
            figure.suptitle(title, y=0.98)  # Move title higher up
        for i, dim_index in enumerate(dims_to_plot):
            # Plot normal time series data
            x_values = x_data if x_data is not None else np.arange(L)
            axes[i].plot(x_values, smooth(y_data[:, dim_index]), linewidth=0.2, label='Normal', color='blue')

            # Highlight anomalies if labels are provided
            if labels_data is not None:
                # Processing one-dimensional label data
                if len(labels_data.shape) == 1:
                    anomaly_indices = np.where(labels_data == 1)[0]
                # Processing two-dimensional label data
                else:
                    anomaly_indices = np.where(labels_data[:, dim_index] == 1)[0]
                
                if len(anomaly_indices) > 0:
                    axes[i].plot(x_values[anomaly_indices], y_data[anomaly_indices, dim_index],
                                 'r.', markersize=2, label='Anomaly')
                    axes[i].legend()

            # Set labels and titles for each subplot
            axes[i].set_title(f'Dimension {dim_index}')
            axes[i].set_xlabel('Timestamp')
            axes[i].set_ylabel(column_names[dim_index])

        return figure
