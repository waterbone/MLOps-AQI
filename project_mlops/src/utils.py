import os
import logging

def setup_logging(log_file="app.log"):
    """
    Setup logging configuration.
    :param log_file: Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def ensure_directory(directory_path):
    """
    Ensure that a directory exists; if not, create it.
    :param directory_path: Path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Directory created at: {directory_path}")
    else:
        logging.info(f"Directory already exists: {directory_path}")

def save_dataframe_to_csv(df, file_path):
    """
    Save a pandas DataFrame to a CSV file.
    :param df: The DataFrame to save.
    :param file_path: The path to save the CSV file.
    """
    df.to_csv(file_path, index=False)
    logging.info(f"DataFrame saved to {file_path}")