"""
=================================================
@Author: Zenon
@Date: 2025-05-18
@Description: File utility functions for handling various file operations.
==================================================
"""
import configparser
import csv
import os
from typing import List, Any, Optional

import yaml  # For YAML processing


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
        >>> data_list = [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 25}]
        >>> write_to_csv('results/data_list.csv', data_list)
        >>> 
        >>> # DataFrame example (assuming pandas is imported as pd)
        >>> # import pandas as pd
        >>> # df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        >>> # write_to_csv('results/data_df.csv', df)
        >>> 
        >>> # String example
        >>> write_to_csv('results/data_str.csv', "This is a line of text")
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
        with open(file_path, mode, newline='', encoding='utf-8') as csvfile:
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
        elif not fieldnames:
            # Handle empty data list case where fieldnames cannot be inferred
            # Depending on desired behavior, could raise error or do nothing
            return  # Or raise ValueError("Fieldnames must be provided for empty list of dicts")


        # Open file and write
        with open(file_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if file doesn't exist or using write mode
            if not file_exists or mode == 'w':
                writer.writeheader()

            # Write all rows
            writer.writerows(data)

    else:
        raise TypeError("Unsupported data type. Supported types: list of dictionaries, DataFrame, or string")


def read_ini_file(file_path: str) -> configparser.ConfigParser:
    """
    Reads an INI configuration file.

    Args:
        file_path (str): The path to the INI file.

    Returns:
        configparser.ConfigParser: A ConfigParser object loaded with data from the INI file.

    Raises:
        FileNotFoundError: If the INI file does not exist.
        configparser.Error: If there is an error parsing the INI file.

    Example:
        >>> # Assume 'config.ini' exists with content:
        >>> # [Section1]
        >>> # key1 = value1
        >>> # try:
        >>> #     config = read_ini_file('config.ini')
        >>> #     value = config.get('Section1', 'key1')
        >>> #     print(value)
        >>> # except FileNotFoundError:
        >>> #     print("INI file not found.")
        >>> # except configparser.Error as e:
        >>> #     print(f"Error reading INI file: {e}")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The INI file was not found: {file_path}")

    config = configparser.ConfigParser()
    try:
        config.read(file_path, encoding='utf-8')
    except configparser.Error as e:
        # Log the error or handle it as needed, then re-raise or raise a custom exception
        # For now, just re-raising the original error
        raise e
    return config


def read_yaml_file(file_path: str) -> Any:
    """
    Reads a YAML configuration file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        Any: The Python object representation of the YAML content (often a dictionary or list).

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.

    Example:
        >>> # Assume 'config.yaml' exists with content:
        >>> # section1:
        >>> #   key1: value1
        >>> # try:
        >>> #     data = read_yaml_file('config.yaml')
        >>> #     value = data.get('section1', {}).get('key1')
        >>> #     print(value)
        >>> # except FileNotFoundError:
        >>> #     print("YAML file not found.")
        >>> # except yaml.YAMLError as e:
        >>> #     print(f"Error reading YAML file: {e}")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The YAML file was not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        # Log the error or handle it as needed, then re-raise or raise a custom exception
        # For now, just re-raising the original error
        raise e
    return data


# Example usage (you can uncomment and run this if you have sample files)
if __name__ == '__main__':
    # Create dummy files for testing
    os.makedirs('results', exist_ok=True)
    os.makedirs('configs', exist_ok=True)

    # Test write_to_csv
    print("Testing write_to_csv...")
    csv_data_list = [{'id': 1, 'value': 'alpha'}, {'id': 2, 'value': 'beta'}]
    write_to_csv('results/test_output.csv', csv_data_list, mode='w')
    write_to_csv('results/test_output.csv', "This is a separator line")
    print(f"CSV file created at results/test_output.csv")

    # Test read_ini_file
    ini_file_path = 'configs/sample.ini'
    with open(ini_file_path, 'w', encoding='utf-8') as f:
        f.write("[DEFAULT]\n")
        f.write("ServerAliveInterval = 45\n")
        f.write("Compression = yes\n")
        f.write("CompressionLevel = 9\n")
        f.write("[user.settings]\n")
        f.write("Port = 50022\n")
        f.write("ForwardX11 = no\n")

    print(f"\nTesting read_ini_file with {ini_file_path}...")
    try:
        ini_config = read_ini_file(ini_file_path)
        print(f"Port from INI: {ini_config.get('user.settings', 'Port')}")
        print(f"Compression from INI (DEFAULT): {ini_config.get('DEFAULT', 'Compression')}")
    except Exception as e:
        print(f"Error during INI test: {e}")

    # Test read_yaml_file
    yaml_file_path = 'configs/sample.yaml'
    yaml_content = {
        'database': {
            'type': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'user': 'admin'
        },
        'logging': {
            'level': 'INFO',
            'file': '/var/log/app.log'
        }
    }
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\nTesting read_yaml_file with {yaml_file_path}...")
    try:
        yaml_data = read_yaml_file(yaml_file_path)
        print(f"Database type from YAML: {yaml_data.get('database', {}).get('type')}")
        print(f"Logging level from YAML: {yaml_data.get('logging', {}).get('level')}")
    except Exception as e:
        print(f"Error during YAML test: {e}")

    # Clean up dummy files (optional)
    # import shutil
    # shutil.rmtree('results')
    # shutil.rmtree('configs')
    # print("\nCleaned up dummy files and directories.")
