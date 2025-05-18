"""
=================================================
@Author: Zenon
@Date: 2025-05-18
@Description: File utility functions for handling various file operations.
==================================================
"""
import csv
import os
from typing import List, Any, Optional


def write_to_csv(file_path: str,
                 data: Any,
                 fieldnames: Optional[List[str]] = None,
                 mode: str = 'a') -> None:
    """
    Write data to CSV file based on data type, supporting dictionary lists, DataFrame and strings.
    
    Args:
        file_path: Path to the CSV file
        data: Data to be written, can be a list of dictionaries, DataFrame or string
        fieldnames: List of field names for CSV headers (optional)
                   If not provided and data is a list of dictionaries, keys from the first dictionary will be used
        mode: File opening mode ('a' for append, 'w' for write)
        
    Returns:
        None
    
    Example:
        >>> # Dictionary list example
        >>> data = [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 25}]
        >>> write_to_csv('results/data.csv', data)
        >>> 
        >>> # DataFrame example
        >>> df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        >>> write_to_csv('results/data.csv', df)
        >>> 
        >>> # String example
        >>> write_to_csv('results/data.csv', "This is a line of text")
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Check if file exists
    file_exists = os.path.isfile(file_path)

    # Process based on data type
    if isinstance(data, str):
        # String type: write directly as a new line
        with open(file_path, mode, newline='') as csvfile:
            csvfile.write(data + '\n')

    elif hasattr(data, 'to_csv') and callable(getattr(data, 'to_csv')):
        # DataFrame type: use pandas built-in to_csv method
        # Write header only if file doesn't exist or using write mode
        header = not file_exists or mode == 'w'
        data.to_csv(file_path, mode=mode, header=header, index=False)

    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        # Dictionary list type: use DictWriter
        # Determine field names if not provided
        if not fieldnames and data:
            fieldnames = list(data[0].keys())

        # Open file and write
        with open(file_path, mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if file doesn't exist or using write mode
            if not file_exists or mode == 'w':
                writer.writeheader()

            # Write all rows
            writer.writerows(data)

    else:
        raise TypeError("Unsupported data type. Supported types: list of dictionaries, DataFrame, or string")
