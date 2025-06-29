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

def merge_train_test_data(train_df: pd.DataFrame, test_df: pd.DataFrame, cols_to_drop: list, target_label: str):
    try:
        log_handle.logger.info("START ...")
        log_handle.logger.info(f'Merging train data shape: {train_df.shape} and test data shape: {test_df.shape}')

        # Create indicator column
        train_df['is_train'] = 1
        test_df['is_train'] = 0

        # Drop specified columns
        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
            test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
            log_handle.logger.info(f'Dropped columns: {cols_to_drop}')

        # Store target variable separately for train data
        train_target = train_df[target_label].copy()
        train_df = train_df.drop(columns=[target_label])

        # Merge dataframes
        merged_df = pd.concat([train_df, test_df], axis=0)
        log_handle.logger.info(f'Merged data shape: {merged_df.shape}')
        log_handle.logger.info("... FINISH")

        return merged_df, train_target
    except Exception as e:
        log_handle.logger.error(f'Failed to merge train and test data: {str(e)}')
        raise
