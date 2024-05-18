
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from mlproject import logger
import joblib


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.preprocessor = None 
        self.transformed_df = None

    def get_data_transformation(self):
        try:
            # Load the dataset
            df = pd.read_csv(self.config.data_path)

            # Dropping specific columns
            cols_to_drop = ['Timestamp', 'Amount Paid', 'Payment Currency']
            df.drop(cols_to_drop, axis=1, inplace=True)

            # Divide the dataset into independent and dependent features
            X = df.drop(columns=["Is Laundering"], axis=1)
            y = df["Is Laundering"]

            logger.info("Dividing the dataset into independent and dependent features completed")

            # Identify numerical and categorical features
            numeric_features = X.select_dtypes(exclude="object").columns
            categorical_features = X.select_dtypes(include="object").columns
            print(f"We have {len(numeric_features)} numeric features & {len(categorical_features)} categorical features")


            # Define the pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler())
                ]
            )

            # Define the pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ]
            )

            # Define the Preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numeric_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ], remainder="passthrough")

            self.preprocessor = preprocessor

            # Transform the whole data using the preprocessor
            X_transformed = preprocessor.fit_transform(X)

            # Get the updated column names after ordinal encoding
            column_names = numeric_features.to_list() + categorical_features.to_list()

            # Oversample the minority class using SMOTE
            oversampler = SMOTE()
            X_resampled, y_resampled = oversampler.fit_resample(X_transformed, y)

            # Combine X_transformed and y back into one DataFrame
            self.transformed_df = pd.DataFrame(X_resampled, columns=column_names)
            self.transformed_df["Is Laundering"] = y_resampled

            logger.info("Data preprocessing completed & Handling Data Imbalance Completed!")

        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e

    def save_preprocessor(self):
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, self.config.preprocessor_path)
            logger.info(f"Preprocessor saved to {self.config.preprocessor_path}")
        else:
            logger.warning("Preprocessor is not available. Please call get_data_transformation to create it.") 

    def train_test_split(self, test_size=0.2, random_state=None):
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not available. Please call get_data_transformation.")

        # Split the data into train and test sets
        train, test = train_test_split(self.transformed_df, test_size=test_size, random_state=random_state)

        # Save the encoded train and test sets in the form of CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split the data into train and test sets.")
        logger.info(f"Shape of train data: {train.shape}")
        logger.info(f"Shape of test data: {test.shape}")
