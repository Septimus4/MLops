"""
Data preparation module for Home Credit Default Risk prediction.
Handles data loading, preprocessing, feature engineering, and train/val/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import logging
import os
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data loading and preprocessing for Home Credit dataset."""

    def __init__(self, data_dir: str = "home-credit-default-risk-DATA"):
        """
        Initialize the data preprocessor.

        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.dataframes = {}
        self.encoders = {}

    def load_data(self):
        """Load all dataset files."""
        logger.info("Loading dataset files...")

        files_to_load = {
            'application_train': 'application_train.csv',
            'application_test': 'application_test.csv',
            'bureau': 'bureau.csv',
            'bureau_balance': 'bureau_balance.csv',
            'previous_application': 'previous_application.csv',
            'POS_CASH_balance': 'POS_CASH_balance.csv',
            'installments_payments': 'installments_payments.csv',
            'credit_card_balance': 'credit_card_balance.csv'
        }

        for name, filename in files_to_load.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                self.dataframes[name] = pd.read_csv(filepath)
                logger.info(f"Loaded {name}: {self.dataframes[name].shape}")
            else:
                logger.warning(f"File {filename} not found")

        return self.dataframes

    def handle_missing_values(self, df: pd.DataFrame, fill_value: float = -999) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.

        Args:
            df: Input dataframe
            fill_value: Value to fill missing values with

        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values for {len(df)} rows")

        # For numeric columns, fill with fill_value
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)

        # For categorical columns, fill with 'UNK'
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('UNK')

        return df

    def encode_categorical_features(self, df: pd.DataFrame, method: str = 'label',
                                  max_categories: int = 30) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            df: Input dataframe
            method: Encoding method ('label' or 'onehot')
            max_categories: Maximum number of categories for one-hot encoding

        Returns:
            DataFrame with encoded categorical features
        """
        logger.info(f"Encoding categorical features using {method} encoding")

        categorical_cols = df.select_dtypes(include=['object']).columns

        if method == 'label':
            for col in categorical_cols:
                if df[col].nunique() <= max_categories:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                else:
                    # For columns with too many categories, create a mapping for most frequent values
                    # and group rare categories together
                    value_counts = df[col].value_counts()
                    top_categories = value_counts.head(max_categories - 1).index.tolist()

                    # Create mapping
                    mapping = {cat: i for i, cat in enumerate(top_categories)}
                    mapping['OTHER'] = max_categories - 1

                    # Apply mapping
                    df[col] = df[col].map(mapping).fillna(max_categories - 1).astype(int)
                    logger.info(f"Encoded {col} with {len(mapping)} categories (including OTHER)")
        elif method == 'onehot':
            # Select categorical columns with reasonable number of categories
            selected_cat_cols = [c for c in categorical_cols if df[c].nunique() <= max_categories]

            if selected_cat_cols:
                enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_features = enc.fit_transform(df[selected_cat_cols])
                encoded_df = pd.DataFrame(
                    encoded_features,
                    columns=enc.get_feature_names_out(selected_cat_cols),
                    index=df.index
                )

                # Drop original categorical columns and add encoded ones
                df = df.drop(columns=selected_cat_cols)
                df = pd.concat([df, encoded_df], axis=1)
                self.encoders['onehot'] = enc

        return df

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic feature engineering.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with additional features
        """
        logger.info("Creating basic features")

        # Create new features dictionary to avoid fragmentation
        new_features = {}

        # Age from DAYS_BIRTH (convert to positive years)
        if 'DAYS_BIRTH' in df.columns:
            new_features['AGE'] = (-df['DAYS_BIRTH'] / 365).astype(int)

        # Employment duration from DAYS_EMPLOYED
        if 'DAYS_EMPLOYED' in df.columns:
            years_employed = (-df['DAYS_EMPLOYED'] / 365).astype(int)
            # Handle anomalous values (365243 days = 1000 years)
            years_employed.loc[years_employed > 100] = -999
            new_features['YEARS_EMPLOYED'] = years_employed

        # Registration duration
        if 'DAYS_REGISTRATION' in df.columns:
            new_features['YEARS_REGISTRATION'] = (-df['DAYS_REGISTRATION'] / 365).astype(int)

        # ID publish duration
        if 'DAYS_ID_PUBLISH' in df.columns:
            new_features['YEARS_ID_PUBLISH'] = (-df['DAYS_ID_PUBLISH'] / 365).astype(int)

        # Create credit-to-income ratio
        if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            new_features['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

        # Create annuity-to-income ratio
        if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            new_features['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

        # Create goods price to credit ratio
        if 'AMT_GOODS_PRICE' in df.columns and 'AMT_CREDIT' in df.columns:
            new_features['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']

        # Add all new features at once to avoid fragmentation
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_features_df], axis=1)

        return df

    def prepare_main_dataset(self, encoding_method: str = 'label', sample_size: Optional[float] = None) -> tuple:
        """
        Prepare the main application dataset for modeling.

        Args:
            encoding_method: Method for encoding categorical features
            sample_size: Fraction of data to sample (0.0-1.0) or None for full dataset

        Returns:
            Tuple of (X, y, feature_names)
        """
        if 'application_train' not in self.dataframes:
            raise ValueError("Application train data not loaded")

        df = self.dataframes['application_train'].copy()

        # Sample data if requested
        if sample_size is not None and 0 < sample_size < 1:
            logger.info(f"Sampling {sample_size:.1%} of the dataset")
            df = df.sample(frac=sample_size, random_state=42).reset_index(drop=True)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Create basic features
        df = self.create_basic_features(df)

        # Encode categorical features
        df = self.encode_categorical_features(df, method=encoding_method)

        # Separate features and target
        if 'TARGET' in df.columns:
            y = df['TARGET']
            X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
        else:
            y = None
            X = df.drop(['SK_ID_CURR'], axis=1, errors='ignore')

        feature_names = X.columns.tolist()

        logger.info(f"Prepared dataset: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y, feature_names

    def create_train_val_test_split(self, X: pd.DataFrame, y: pd.Series,
                                  test_size: float = 0.2, val_size: float = 0.2,
                                  random_state: int = 42) -> tuple:
        """
        Create train/validation/test splits.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining data)
            random_state: Random state for reproducibility

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Creating train/validation/test splits")

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )

        logger.info(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_feature_importance_baseline(self, X: pd.DataFrame, y: pd.Series,
                                      n_estimators: int = 50) -> pd.DataFrame:
        """
        Get baseline feature importance using Random Forest.

        Args:
            X: Feature matrix
            y: Target vector
            n_estimators: Number of trees in Random Forest

        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier

        logger.info("Computing baseline feature importance")

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_leaf=4,
            max_features=0.5,
            random_state=2018,
            n_jobs=-1
        )

        rf.fit(X, y)

        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance_df