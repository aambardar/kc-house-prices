# data_loader.py
import pandas as pd
import base_utils_logging as log_handle

def load_data(data_file_path):
    try:
        log_handle.logger.info("START ...")
        log_handle.logger.info(f'Loading data from path: {data_file_path}')
        df_data = pd.read_csv(data_file_path)
        log_handle.logger.info(f'Loaded train data with shape: {df_data.shape}')
        log_handle.logger.info(f'Successfully loaded data from path: {data_file_path}')
        log_handle.logger.info("... FINISH")
        return df_data
    except Exception as e:
        log_handle.logger.error(f'Failed to load data from {data_file_path}: {str(e)}')
        raise

def refactor_col_names(df: pd.DataFrame):
    log_handle.logger.info("START ...")
    df.columns = df.columns.str.replace(' ', '')
    df = df.rename(columns={'YearRemod/Add': 'YearRemodAdd'})
    log_handle.logger.info("... FINISH")
    return df