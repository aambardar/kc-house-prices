"""
Main script for House Prices Prediction
This script replaces the Jupyter notebook functionality in a modular way
"""

import os
import numpy as np
import pandas as pd
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import comet_ml

# Local imports
import proj_configs as configs
import proj_utils as utils
import proj_utils_data_loader as data_loader
import proj_utils_feat_engg as feat_engg
import proj_utils_model as model_utils
import proj_utils_plots as plot_utils
import base_utils_logging as log_utils


class HousePricePredictor:
    def __init__(self):
        self.setup_environment()
        self.logger = log_utils.setup_logging()

    def setup_environment(self):
        """Initialize environment settings and configurations"""
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("mlflow").setLevel(logging.ERROR)

    def load_data(self):
        """Load and prepare the training and test datasets"""
        self.df_raw_train = data_loader.load_data(configs.TRAIN_FILE)
        self.df_raw_test = data_loader.load_data(configs.TEST_FILE)

        # Define column categories
        self.insignificant_cols = ['Id']
        self.target_col = 'SalePrice'
        self.ignorables_cols = self.insignificant_cols + [self.target_col]
        self.ordinal_cols = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual',
                             'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual',
                             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                             'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
                             'GarageCond', 'PoolQC']
        self.temporal_cols_name_pattern = ['Yr', 'Year']

    def prepare_data(self):
        """Prepare and process the data for modeling"""
        # Merge train and test data
        self.df_raw_all, self.df_raw_target = data_loader.merge_train_test_data(
            self.df_raw_train,
            self.df_raw_test,
            self.insignificant_cols,
            self.target_col
        )

        # Split into train and test
        self.df_train = self.df_raw_all[self.df_raw_all['is_train'] == 1].iloc[:, :-1]
        self.df_test = self.df_raw_all[self.df_raw_all['is_train'] == 0].iloc[:, :-1]

        # Classify features
        self.feature_categories = feat_engg.classify_columns(
            df=self.df_train,
            n_cat_threshold=configs.CATEGORICAL_CARDINALITY_THRESHOLD_ABS,
            threshold_type='ABS',
            cols_to_ignore=self.ignorables_cols,
            temporal_cols_name_pattern=self.temporal_cols_name_pattern,
            ordinal_cols=self.ordinal_cols
        )

        # Get column categories
        (self.cols_num_continuous, _, self.cols_num_discrete, _,
         self.cols_cat_nominal, _, self.cols_cat_ordinal, _,
         self.cols_object, _, self.cols_temporal, _,
         self.cols_binary, _) = feat_engg.get_cols_as_tuple(self.feature_categories)

    def create_train_val_split(self):
        """Create training and validation splits"""
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.df_train,
            self.df_raw_target,
            test_size=configs.VALIDATION_SIZE,
            random_state=configs.RANDOM_STATE
        )

    def setup_preprocessing(self):
        """Setup the preprocessing pipeline"""
        self.num_columns = self.cols_num_continuous
        self.cat_columns = (self.cols_cat_nominal + self.cols_cat_ordinal +
                            self.cols_num_discrete + self.cols_binary + self.cols_object)
        self.tempo_columns = self.cols_temporal

        self.pproc_pipe = feat_engg.create_pproc_pipeline(
            self.num_columns,
            self.cat_columns,
            self.tempo_columns
        )

    def train_model(self):
        """Train the model using XGBoost with hyperparameter optimization"""
        comet_experiment = comet_ml.Experiment()
        run_name = 'xgb-house-prices'

        try:
            self.optimized_study = model_utils.run_hyperparam_tuning_xgb_exp(
                self.X_train_transformed,
                self.y_train_transformed,
                self.X_val_transformed,
                self.y_val_transformed,
                comet_experiment,
                run_name,
                configs.OPTUNA_TRIAL_COUNT
            )
        finally:
            comet_experiment.end()

    def transform_data(self):
        """Transform the data using the preprocessing pipeline"""
        self.X_train_transformed = self.pproc_pipe.fit_transform(self.X_train)
        self.X_val_transformed = self.pproc_pipe.transform(self.X_val)
        self.y_train_transformed = self.y_train.to_numpy()
        self.y_val_transformed = self.y_val.to_numpy()

    def make_predictions(self):
        """Make predictions on test data"""
        self.test_transformed = self.pproc_pipe.transform(self.df_test)
        self.test_predictions = self.model.predict(self.test_transformed)

    def save_predictions(self):
        """Save predictions to a CSV file"""
        submission_df = pd.DataFrame({
            'Id': self.df_raw_test.Id,
            'SalePrice': self.test_predictions
        })

        submission_file = os.path.join(
            configs.PATH_OUT_SUBMISSIONS,
            'submission.csv'
        )
        submission_df.to_csv(submission_file, index=False)
        self.logger.info(f"Predictions saved to {submission_file}")

    def evaluate_model(self):
        """Evaluate model performance"""
        train_preds = self.model.predict(self.X_train_transformed)
        val_preds = self.model.predict(self.X_val_transformed)

        train_mse = round(mean_squared_error(self.y_train_transformed, train_preds), 5)
        val_mse = round(mean_squared_error(self.y_val_transformed, val_preds), 5)
        train_r2 = round(r2_score(self.y_train_transformed, train_preds), 5)
        val_r2 = round(r2_score(self.y_val_transformed, val_preds), 5)

        self.logger.info("=== Model Performance ===")
        self.logger.info(f"Train MSE: {train_mse}, Train R2: {train_r2}")
        self.logger.info(f"Validation MSE: {val_mse}, Validation R2: {val_r2}")

        metrics_string = (f'=== Model Performance === \n'
                          f'Train MSE: {train_mse}, Train R2: {train_r2} \n'
                          f'Validation MSE: {val_mse}, Validation R2: {val_r2}')

        utils.save_file('metrics', 'validation_metrics.txt',
                        configs.PATH_OUT_MODELS, metrics_string)


def main():
    predictor = HousePricePredictor()

    # Pipeline execution
    predictor.load_data()
    predictor.prepare_data()
    predictor.create_train_val_split()
    predictor.setup_preprocessing()
    predictor.transform_data()
    predictor.train_model()
    predictor.make_predictions()
    predictor.save_predictions()
    predictor.evaluate_model()


if __name__ == "__main__":
    main()